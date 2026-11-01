# DSpark: Fast- und Strict-Modus

DeepSeek-V4.1-Flash-1017 mit optionalem DSpark-Sidecar. Ohne `--mtp-sidecar`
bleiben Sidecargewichte, Features und Draftgraphen ausgeschaltet. Hauptgraph und
Haupt-KV behalten 40 Layer. Konvertierung: [dsv41-mtp.md](dsv41-mtp.md).

```text
--mtp-sidecar /daten/inferssd-models/DeepSeek-V4.1-Flash-1017-MTP/DeepSeek-V4.1-Flash-1017-MTP-Q8_0.gguf
--mtp-mode fast --mtp-drafts 1
--ssd-vram-reserve-mib 512 --ssd-vram-margin-mib 576
```

`fast` ist mit Sidecar der Standard; `--mtp-drafts` erlaubt 1..4 (Standard: 1).
Der normale automatische VRAM-Expert-Tier ist zulässig. Der bisherige strenge
Prüfpfad bleibt mit `--mtp-mode strict --ssd-vram-budget 0` verfügbar; er verwendet
weiter einen Draft, einen vollständigen exakten Snapshot und sequentielle
Zielentscheidungen. Dense und Attention bleiben auch dort auf der GPU.

## Draft und Ziel-Sampling

Vor den Hauptlayern 37/38/39 werden die HC-Mittelwerte als 15.360 Werte pro
Position ausgegeben, auch für Eingaben ohne angeforderte Logits. DSpark liest die
letzten 128 committed Features in physischer Ringreihenfolge. Projektion/Norm,
drei DSpark-Blöcke, verzögerte HC-Mixes, RoPE, Attention-Sinks, Shared Experts und
Top-3/128 folgen der lokalen offiziellen Referenz. Alle fünf internen Positionen
(bekanntes Token plus vier Noise-Positionen) bleiben bidirektional sichtbar.
Explizite FP8/E8M0-Quant-/Dequant-Stellen bleiben erhalten.

Die ersten k Outputpositionen verwenden den gemeinsamen Haupt-Head und einen
verketteten Markov-Bias: Der Argmax einer Position bestimmt das Markov-Embedding
der nächsten. Ein Graph liefert alle Vorschläge; Drafting verbraucht keinen
Ziel-RNG. Die optionale Confidence-Projektion wird weiterhin nicht ausgewertet.

Fast verifiziert `[bekanntes Token, Draft 1, ..., Draft k]` in einer tatsächlichen
Mehrtoken-UBatch. Pro Verifierzeile läuft der normale Ziel-Sampler einschließlich
Temperatur, Penalties und Grammar. Das mit dem Draft übereinstimmende Präfix wird
akzeptiert; die erste Abweichung oder das Bonus-Token wird ausgegeben. EOG beendet
die Entscheidung. Es gibt keine sequentielle Neuentscheidung.

Für einen deterministischen Argmax-Draft entspricht dies spekulativem Sampling
mit einer Punktverteilung als Vorschlag: Die Zielprobe entscheidet sowohl über
Akzeptanz als auch über das Korrekturtoken. Jede ausgegebene Probe stammt aus der
normalen Zielverteilung des gültigen Präfixes. Gleiche Seeds müssen bei anderer
Batch-Arithmetik/Expert-Platzierung keine identischen Tokens ergeben.
Wahrscheinlichkeiten werden vor Rollback/Replay für jedes Ziel-Sample gesichert.

## Begrenzter Undo

Fast nimmt für die Verifiertransaktion keinen vollständigen Sequenz-Snapshot. Die Transaktion erlaubt eine
Single-Stream-Append-UBatch mit höchstens fünf Eingaben. Sie sichert:

- Physische Raw-K-Schreibzeilen und die von `apply_ubatch` veränderten Zellen,
  Token-Metadaten und Schreibköpfe beider Raw-Caches.
- Die tatsächlich geplanten CSA-/HCA-/Indexer-Schreibziele einschließlich
  Dummyzeilen. V4.1 speichert beide Ratios im CSA-Speicher; Indexer-Zeilen nutzen
  denselben jeweiligen Plan wie ihre KV-Quelle.
- Die beschriebenen KV-/Score-Ringzeilen der Compressor-Zustände und `rs_idx`.

Zusammenhängende Zeilen werden gemeinsam kopiert; die Datenmenge hängt von der
Batchgröße und den Modellformen ab, nicht von der gesamten Kontextlänge.
Vor GPU-Lesen/Restore wird synchronisiert. Ein zweiter Begin, eine zusätzliche
UBatch oder eine nicht unterstützte Sequenzgeometrie wird abgelehnt.

Bei vollständiger Akzeptanz werden nur die Undo-Daten freigegeben. Bei Reject
werden die gesicherten Zeilen restauriert und das bereits gewählte Eingabepräfix
tokenweise erneut berechnet, ohne Sampling oder RNG-Verbrauch.
Ein generisches DSv4-Suffix-Trim erfolgt danach nicht. Die Featurehistorie wird
ab der Replayposition ersetzt; ein zusätzliches Fenster erhält ältere committed
Features. Ein Promptcache-Treffer ohne vollständige Features fällt vorübergehend
auf normales Decode zurück.

Strict verwendet `LLAMA_STATE_SEQ_FLAGS_EXACT_LAYOUT`, das alle Raw-Zellen,
physischen Indizes, Heads und komprimierten Zustände erhält. Das gewöhnliche
Serializerformat bleibt unverändert.

## Ausführung und Grenzen

Der Prefill-Lifecycle reserviert für DSpark dauerhaft fünf Decode-Eingaben und
Outputs. Eine erneute Ein-Token-Reserve würde bei jeder Verifierbatch den
Scheduler neu erzeugen und den Expert-Cache leeren. Innerhalb einer Decode-Phase
bleibt außerdem der größte CUDA-Scratch-Pool reserviert: Ein kleinerer Draftgraph
darf den Arbeitsspeicher des nächsten Verifiers nicht dem Expert-Tier überlassen.

Kleine Decode-Batches verwenden den vorhandenen tierfähigen Vektorexecutor pro
Expert-Zeile innerhalb eines gemeinsamen Dense-/Attention-Graphen. Sie eröffnen
keine große Prefill-Phase. Ein fortlaufender Expert-Mikroschrittzähler vermeidet
Cacheleerungen durch wiederholte oder übersprungene Modellpositionen. Der
`count_decay2`-Cache kann eine Layergruppe mehrfach innerhalb desselben Schritts
bedienen. First-Row-Prefetch verwendet die vorhandene Tiefeneinstellung.

F32-Anforderungen kleiner Float-/BF16-Matrizen nutzen den vorhandenen
Mehrspalten-Vektorkernel. Insbesondere Router, Kompressionsprojektionen und
Output-Head sollen F32-Aktivierungen nicht nur wegen einer kleinen Batch zu BF16
runden. Das ist keine universelle Bitparitätszusage.

Unterstützt: Text, ein Slot, gemeinsamer SSD-/RAM-/VRAM-Expert-Runtime und
konfigurierte UBatch mindestens fünf. Separate Draftmodelle und LoRA werden mit
Sidecar abgelehnt. Der benachbarte `mtp-experts.tsv` wird dem Hauptindex
hinzugefügt. Draft-Layer 40–42 haben 128 Experten/Top-3; Hauptlayer bleiben
384/Top-6. Der Loader prüft Format, Dateibereiche und alle 81 Tensorformen/-typen;
ein kryptografischer Hauptcheckpoint-Fingerprint ist noch nicht enthalten.

## Prüfungen

```bash
cmake --build build-linux-cuda --parallel 16 --target \
  test-dsv4-state test-dspark-quant test-ssd-settings-telemetry test-dspark-state-model
ctest --test-dir build-linux-cuda \
  -R '^(test-dsv4-state|test-dspark-quant|ssd_settings_telemetry)$' --output-on-failure
```

`test-dsv4-state` prüft beide Serializerformate/Ratiogruppen und gezieltes
Überschreiben der geplanten Undo-Zeilen bei k=1..4, einschließlich
127/128/129 und 255/256/257. Die Same-Step-Erweiterung der CPU-Sink-Aktivierungs-
Fixture verlangt RAM-Hits ohne SSD-Reads und bitgleiche Ergebnisse.

`DSPARK_UNDO_CANARY=1 test-dspark-state-model ...` führt den Realmodell-Canary aus.
Für 127/128/129, 255/256/257 und 4104 Positionen werden alle Batchbreiten 2..5 und
alle behaltenen Präfixe geprüft. Nach Reject/Undo/Replay werden Folgelogits mit
frischen sequentiellen Referenzen und deren Wiederholung verglichen. Full Accept
wird zusätzlich gegen frische identische Batches geprüft; Batch-/Einzel-
Numerik wird separat protokolliert. Vor dem Replay müssen die serialisierten
Cachebytes nach Delta-Undo exakt dem Ausgangszustand entsprechen. Full Accept
darf beim Verwerfen des Undo keine Cachebytes verändern.

Der Auto-Tier verändert durch CPU-/GPU-Platzierung die Rohlogits auch ohne Undo.
Ein unabhängiger Kontrolllauf führt dieselbe vergiftende Batch aus und verwendet
den vollständigen exakten Restore. Die Forward-Toleranz beträgt das Dreifache
der gemessenen Kontrollabweichung, mindestens 0,02 im Mittel / 0,25 maximal bei
Rohlogits. Full Accept verwendet vier unabhängige frische Batchkontrollen, um
kalte und warme Expert-Platzierung abzudecken. Die bitgenauen Cacheprüfungen
bleiben davon unabhängig. Das 4K-Akzeptanzkriterium für **Logprobabilities** bleibt
unverändert: Präfix mindestens 150 Tokens und mittleres |Δlogprob| unter 0,02. Ohne die Umgebungsvariable bleibt der
vollständige exakte Serializer-Canary aus Paket I verfügbar.

`tests/dspark_equivalence.py` zeichnet Greedy-, Sampling-/Penalty- und Grammar-
Fälle auf. `tests/dspark_sampling.py` prüft echte kategoriale Samples über viele
Seeds mit einem festen Grammar-Präfix, das die Zufallsentscheidung in einen
spekulativen Schritt verlegt. Der offizielle 4K-Vergleich und Messwerte stehen
im Ergebnis-J-Protokoll; Fast ist kein zugesicherter Beschleunigungsfaktor.

## Messstand 1. November 2026

RTX PRO 4500 Blackwell, CUDA 13.2, Threads 12, Prefetch 4, Auto-VRAM,
Reserve/Margin 512/576 MiB; offizieller 4K-/256-Token-Test, frischer Server je Lauf.
Alle folgenden Off/On-Vergleiche verwenden denselben Messbuild J_11.

| Modus | Decode letzte 128 tok/s | Akzeptanz | Tokens/Spec-Schritt | Vergleich |
|---|---:|---:|---:|---|
| Ohne Sidecar | 8,39 | — | — | Referenz |
| Fast k=1 | 6,71 | 84,78 % | 1,848 | 256 gleiche Tokens, Δlogprob 0,00909 |
| Fast k=2 | 5,86 | 68,69 % | 2,374 | PRÜFEN: Präfix 33 |
| Fast k=3 | 6,05 | 63,98 % | 2,920 | PRÜFEN: Präfix 105 |
| Fast k=4 | 5,49 | 57,65 % | 3,299 | 256 gleiche Tokens, Δlogprob 0,01093 |

Zwei weitere k=1-Läufe ergeben 6,86 und 6,79 tok/s, beide GLEICHWERTIG
(Präfix 228/256; mittlere Δlogprob 0,01017/0,01057). Mittelwert k=1:
**6,78 ± 0,07 tok/s (Stichprobenstreuung, n=3), etwa 19 % langsamer als Off**.
Deshalb wurde der abschließende CLI-Standard auf k=1 gesetzt. Der Inferenzpfad
der Messreihe bleibt dabei unverändert; J_12 prüft den Default zusätzlich.

Draft-Median k=1: 22,65 ms statt der ungefähr 64 ms aus Paket I. Der Tensor-Undo
umfasst 125.184 Byte plus kleine Metadaten und braucht median 0,41 ms. Verifier,
Transport und Graphwechsel verbleiben bei ungefähr 248 ms pro Zwei-Token-Block;
die Hauptmodell-Experten werden weiterhin zeilenweise berechnet. Reject-Replay
addiert weitere Arbeit. Der bisherige gruppierte Prefill-Expert-Pfad war für die
kleinen Batches deutlich langsamer.

CPU-Expert-Canary: 70 Reject-Folgelogitvergleiche exakt, 28 frische Full-Accept-
Batchvergleiche exakt. Auto-Tier-Canary: 70 bytegenaue Restores, 28 bytegenau
unveränderte Full-Accept-Discards; Forward-Vergleiche innerhalb des unabhängig
ermittelten Kontrollbereichs. Jeweils alle geforderten Grenzen und k=1..4.

64 kategoriale Seeds pro Modus bestehen die Verteilungssanity: A/B/C/D Off
13/18/27/6, Fast 14/20/27/3, empirische Totalvariation 0,046875. Grammar/JSON
ist 13/13 gleich. Der zusätzliche freie Greedy-Prompt weicht dagegen ab Token 9,
der Sampling-/Penalty-Fall ab Token 17 ab. **Allgemeine Greedy-Tokenparität und
ein Produktions-Speedup sind nicht nachgewiesen.** Für die höchste gemessene
Produktionsgeschwindigkeit bleibt der Sidecar ausgeschaltet.
