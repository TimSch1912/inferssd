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
Ziel-RNG. `--mtp-confidence P` wertet optional den originalen Confidence-Head aus:
F32-Projektion von `[hidden, Markov-Embedding]`, danach Sigmoid. Ab dem zweiten
Vorschlag wird am ersten Score unter P abgeschnitten; mindestens ein Draft bleibt.
P=0 (Standard) deaktiviert das Gate. Der Score ist eine unkalibrierte Heuristik,
keine nachgewiesene Akzeptanzwahrscheinlichkeit. Strict ignoriert das Gate.

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
in einer einzigen Batch erneut berechnet, ohne Sampling oder RNG-Verbrauch.
Das benutzt denselben Mehrspalten-Decode wie der Verifier; bei nur einer
behaltenen Eingabe bleibt es der normale Einzeltoken-Pfad.
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

Decode-Batches mit 2..5 Eingaben planen pro Layer die Expert-Union einmal. Der
CPU-Sink liest/admittiert jedes Gewichts-Tripel einmal und verwendet die bestehenden
MXFP4-2x/4x-Kernel für die zugehörigen Spalten. RAM-Hits und fertige SSD-Reads
werden während weiterer Reads berechnet. Die CUDA-Stufe pinnt alle residenten
Experten der Union und startet ihre Up/Gate-, Aktivierungs-, Q8- und Down-Kernel
mit nur einem H2D/D2H-Paar pro Layer. DP4A verarbeitet benachbarte Gewichtsblöcke
parallel; die Float-Terme werden in der bisherigen GPU-Reihenfolge addiert.
Die bestehenden Batch-Puffer reichen auch für Deskriptoren: keine zusätzliche
GPU-Scratch-Reservierung. Routing-Gewichte und Summen bleiben in ursprünglicher
Slotreihenfolge, einschließlich doppelter IDs.

Es gibt keinen Prefill-Phasenwechsel in diesem Pfad. Ein fortlaufender
Expert-Mikroschrittzähler hält die Decode-Caches gültig; die Cache-Hitze zählt
jede Token-Auswahl eines Union-Experten. Der Next-Layer-Prefetch enthält die
vorhergesagte Union aller Eingaben, dedupliziert und nach Router-Rang verflochten.
Der normale Einzeltoken-Executor bleibt separat. Rohwahrscheinlichkeiten im
Fast-Verifier sortieren nur die angeforderten Top-N; der Softmax-Nenner umfasst
weiter alle Vokabulareinträge. Der Ziel-Sampler wird dadurch nicht verändert.

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
frischen, vollständig restaurierten Präfix-Batches und deren Wiederholung verglichen. Full Accept
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

Die Decode-Union-Fixture in `test-kimi-cpu-sink --activation-tests` prüft 2..5
Spalten bitgenau gegen CPU-Einzeltoken-Ausführung, Cold-Reads pro Expert-Matrix,
Warm-Hits ohne Reads, doppelte/negative IDs und leere Union. Sie läuft auch ohne
Ready-first. `test-kimi-mxfp4-chain --decode-only` prüft 1..5 Spalten und drei
parallele Experten bitgenau gegen den bisherigen CUDA-Reihenkernel, plus ungültige
Deskriptoren. GPU-Tests nur mit exklusiv übernommener GPU ausführen.

## Messstand K (1. November 2026)

Der Mehrspalten-Decode ist implementiert; das Ziel von etwa 150 ms für zwei
Verifier-Eingaben und mehr als 11 tok/s wurde nicht erreicht. Die aus der
Suchmatrix ausgewählte Konfiguration ist k=3, `-t 8 -tb 8`, Prefetch 4 und
`--mtp-confidence 0`. Drei finale Läufe desselben Builds K10 liefern
7,726 / 7,985 / 7,802 tok/s letzte128: **7,84 ± 0,13 tok/s** (Stichprobenstreuung,
n=3). Die beiden vollständigen Off-Kontrollen desselben Builds liefern
8,408 / 8,444 tok/s. Alle drei MTP-Läufe sind nach unverändertem `vergleich.py`
GLEICHWERTIG (Präfix 228/228/256; mean Δlogprob 0,01031/0,00977/0,00961).
Für die höchste gemessene 4K-Geschwindigkeit bleibt Produktion ohne Sidecar.

Die vollständige k-Matrix auf K8/Off8 (12 Threads, Prefetch 4) ist ebenfalls
GLEICHWERTIG: Off 8,390; k1 7,993; k2 7,496; k3 8,070; k4 7,159 tok/s.
Confidence 0,5 erreicht dort 7,885; Confidence 0,8 auf K9 mit acht Threads
8,101 tok/s. Der höchste Suchlauf ist k3/T8/P4 mit 8,166 tok/s; er ersetzt
nicht die genannten drei Abschlusswiederholungen.

Im warmen k1-Verifier liegen median 205,6 ms für Hauptgraph/Undo/Transport an,
gegenüber ungefähr 248 ms in J. Zwei Tokens wählen im Mittel 10,36 verschiedene
Experten pro Layer, gegenüber 12 getrennten Expert-Durchläufen: nur 13,7 %
weniger Gewichtsdurchläufe. Die SSD-Bedarfsreads bleiben median 969 MB pro Block,
vor zusätzlichem Prefetch. Top-N-Teilsortierung senkt die Accept-/Rohprob-Ausgabe
von ungefähr 15 auf 1,3 ms. Ein kleiner Draftgraph wird weiterhin aufgebaut;
Aufbau und Scheduler-Allokation liegen jeweils deutlich unter einer Millisekunde.

CPU-Canary mit Batch-Präfix-Replay: 70 bytegenaue Restores, 28 unveränderte
Full-Accept-Discards und alle Folgelogits exakt gegen frische Präfix-Batches.
Auto-VRAM: dieselben 70/28 Byteprüfungen und alle Forward-Vergleiche innerhalb
des unabhängigen Kontrollbereichs. Alle Positionen und Breiten wie oben.
Die Rohlogit-Kontrollgrenzen bleiben unverändert; das 4K-Logprob-Kriterium ebenso.
Sechs relevante CTests, die vollständige CPU-Aktivierungs-Fixture, der direkte
CUDA-Bitvergleich und der Mock-Tier-Prefetch-Test bestehen. Der 80-Experten-
Tabellentest prüft Heat-Multiplizität, Ready-Leases und Masken über Bit 31 hinaus.

64 kategoriale Seeds pro Modus: Off A/B/C/D=12/20/28/4, MTP=13/17/31/3,
Totalvariation 0,0625, Sanity bestanden. 323/324 Post-Prob-Zeilen sowie die
Rohprob-Zeilen der freien Tests sind gültig. Freier Greedy96 divergiert ab
Token 11, Sampling96 ab 17; Grammar13 ist gleich. Das ist keine allgemeine
Greedy-Paritäts- oder universelle Verteilungszusage. CPU-Off/Strict bleibt bei
16/16/13 Tokens einschließlich sämtlicher Post-Probs exakt. Stream-Abbruch nach
drei Tokens mit anschließender vollständiger 16-Token-/16-Probs-Anfrage besteht.
Alle Optimierungsmessungen verwenden Kontext 8192, nicht den langen
Produktionskontext 524288. Änderungen sind lokal, nicht installiert/gepusht.
