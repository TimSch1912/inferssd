# DSpark-Laufzeit: erster Korrektheitsmodus

DeepSeek-V4.1-Flash-1017, optional mit `--mtp-sidecar PATH`. Ohne diese Option
werden keine Sidecargewichte, Features oder Drafts erzeugt. Der Hauptgraph und
seine KV-Caches behalten 40 Layer. Der Entwurf und die Konvertierung stehen in
[dsv41-mtp.md](dsv41-mtp.md).

Der Sidecar benötigt den benachbarten `mtp-experts.tsv`. Dessen Einträge werden
mit dem vorhandenen Hauptindex in **einem** SSD-/RAM-/VRAM-Tier registriert.
Globale Draft-Layer sind 40–42, mit je 128 Experten und Top-3; Hauptlayer
behalten 384/Top-6. Experten mit gleicher Payload-Geometrie dürfen einen kleineren
Schlüsselbereich im bereits reservierten Tier verwenden. Die tensorweise Grenze
bleibt verbindlich; Expert 128 ist für einen Draft-Tensor ungültig.

Der Loader prüft Format, vollständige Dateibereiche, Konfiguration und alle 81
Tensorformen und -typen gegen Hauptlayer-Analoga bzw. die sechs Sondergewichte.
Die Bindung an das Hauptmodell wird anhand von Architektur, Dimensionen und
Vokabulargröße geprüft; eine kryptografische Bindung der Hauptgewichte ist noch
nicht enthalten. Der Sidecar hat keinen eigenen Tokenizer und verwendet die
beiden unterschiedlichen Hauptgewichte für Embedding und Ausgabe.

## Graph und Transaktion

Vor Hauptlayer 37/38/39 werden die HC-Mittelwerte in dieser Reihenfolge als
15.360 Werte ausgegeben. Der Transport berücksichtigt diese Breite an allen
Allokations-, Kopier- und Getterstellen und erfasst auch Eingaben ohne Logitausgabe.

Der Drafter liest die letzten 128 committed Features. Er projiziert und
normalisiert sie mit `mtp.0.main_proj/main_norm`, baut daraus die Kontext-KV
aller drei Stufen und verarbeitet das bekannte Folgetoken plus vier Noise-IDs.
Alle fünf internen Positionen bleiben bidirektional sichtbar. HC-Pre-Mixes wirken
wie im Referenzcode eine Sublayer verzögert. Dazu kommen Attention-Sinks,
positionsrichtige RoPE/inverse RoPE, Shared Expert, Top-3/128 mit Clamp 10 und
Scale 1,5 sowie der gemeinsame Head mit Markov-Bias. Die drei expliziten
FP8/E8M0-Quant/Dequant-Stellen der Attention sind enthalten.

Referenz: [model.py, Revision 1bae0e1088993b9b82de08f59de5be5a34e42170](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py).
Die Gewichte bleiben im konvertierten Q8_0/MXFP4/BF16-Format und verwenden die
vorhandenen Engine-Rechenpfade. Dies ist keine Zusage bitgleicher PyTorch-/FP8-
Inferenz. Der erste Meilenstein verwendet genau einen Argmax-Vorschlag; die
anderen Output-Heads und Confidence-Auswertung werden nicht berechnet. Ihre
Hidden-Positionen nehmen weiterhin an allen drei Blöcken teil.

Vor `[bekanntes Token, Draft]` entsteht ein vollständiger Sequenz-Snapshot mit
`LLAMA_STATE_SEQ_FLAGS_EXACT_LAYOUT`. Er erhält sämtliche Raw-Zellen, deren
physische Indizes und die Schreibköpfe. Ein reiner SWA-Ausschnitt mit anschließend
neu belegten Zellen reicht für einen bitgleichen Folgeschritt nicht aus; der
Realmodell-Canary deckt insbesondere die Grenze 128/129 ab. Das gewöhnliche
Snapshotformat ohne dieses Flag bleibt unverändert.
Der Hauptgraph berechnet beide Logitzeilen gemeinsam. Der Server restauriert
anschließend **vor** Ziel-Sampling und RNG-Verbrauch und entscheidet mit
sequentiellen Ziel-Logits. Ein Reject replayt nur die bekannte Eingabe; ein
Accept zusätzlich den akzeptierten Draft und zieht dann das Bonus-Token.
EOG beendet die Entscheidung ohne weiteres Decode. Wahrscheinlichkeiten werden
bei jeder Zielentscheidung gesichert, bevor das nächste Sample sie überschreibt.
Der bereits ausgegebene bekannte Token wird nicht erneut gesampelt.

Es erfolgt kein DSv4-Suffix-Trim nach diesem vollständigen Restore. Auch die
Featurehistorie wird beim Replay ab dessen Position ersetzt; ein zusätzliches
Fenster verhindert, dass Verifierzeilen noch benötigte committed Features
verdrängen. Beim Requestwechsel wird die Historie geleert. Nach einem Prompt-
Cachetreffer ohne vollständige Features fällt der Server auf gewöhnliches Decode
zurück, bis wieder ein vollständiges Fenster vorhanden ist.

## Grenzen und Tests

Dieser Modus ist absichtlich langsam: Kontext-KV werden aus Features neu gebildet,
Graphen wechseln, die vollständige History wird gesichert und Zielentscheidungen
werden sequentiell wiederholt. Ein Batch-Fastpath ist erst nach belastbarer
numerischer Parität zulässig. `DSpark verifier logit audit` protokolliert den
Bytevergleich; `speculative block audit` zählt Vorschläge, Accepts und Zeiten.

Unterstützt wird zunächst Text, ein Slot, SSD-Ausführung und eine UBatch von
mindestens fünf Positionen. Separate Draftmodelle und LoRA werden abgelehnt.
Mehrere Slots, multimodale Drafts, begrenzte Undo-Snapshots, persistente Draft-KV,
mehrere Vorschläge und Confidence-Steuerung bleiben weitere Arbeit.

Modellfreie Checks:

```bash
cmake --build build-linux-cuda --parallel 16 --target test-dsv4-state test-dspark-quant test-ssd-settings-telemetry
ctest --test-dir build-linux-cuda -R '^(test-dsv4-state|test-dspark-quant|ssd_settings_telemetry)$' --output-on-failure
```

`test-kimi-vram-format-compat` benötigt eine ausdrücklich verfügbare CUDA-GPU und
prüft auch den Wechsel auf kleinere Expertmengen ohne neuen Cache/Engine.
`test-dspark-state-model` ist ein manuell zu startender Realmodell-Canary. Mit
festen CPU-Experten (`--ssd-vram-budget 0`) vergleicht er Folgelogits vor/nach
Restore sowie nach einer verworfenen Zwei-Token-Batch an 127/128/129,
255/256/257 und 4104 Positionen byteweise. Er benötigt Modell-/SSD-Argumente wie
der Server und dieselbe Memory-cgroup.

`tests/dspark_equivalence.py` zeichnet echte Token-IDs für Greedy, Fixed Seed
mit Penalty und Grammar auf und vergleicht zwei Serverstarts exakt. Reale
Server-/Benchmarkergebnisse stehen im zugehörigen Ergebnis-I-Laufprotokoll.
