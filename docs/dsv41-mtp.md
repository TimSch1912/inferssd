# DeepSeek-V4.1-1017: DSpark-Sidecar und MTP-Entwurf

Stand: 31.10.2026, Worktree `/home/timo/work/inferssd-opt-mtp`, Branch `opt-mtp`, Ausgangsstand `e45e752`. D/G wurden zuerst gelesen. Konvertierung und Dateiprüfungen erfolgten CPU-only; kein Serverstart, keine Modellinferenz, kein GPU-Test. Die folgenden Loader-/Graph-/Server-Änderungen sind **Entwurf**, noch keine implementierte Laufzeitfunktion.

## Ergebnis und notwendige Korrektur des ersten Meilensteins

Der Sidecar liegt unter:

```text
/daten/inferssd-models/DeepSeek-V4.1-Flash-1017-MTP/DeepSeek-V4.1-Flash-1017-MTP-Q8_0.gguf
```

Er enthält 81 Tensoren aus allen 2.401 `mtp.*`-Quelltensoren, 7.967.715.360 Byte. Typen: **9 MXFP4, 25 Q8_0, 6 BF16, 41 F32**. Die 9 Expert-Tensoren enthalten zusammen 7.219.445.760 Byte; die übrigen Gewichte ungefähr 714 MiB. Manifest mit Quellnamen, logischen Formen in NumPy-Reihenfolge, Typen, Analogon und SHA256 je Ausgabetensor: gleichnamige `.manifest.json`. Der zusätzliche SSD-Index heißt `mtp-experts.tsv` (1.152 Einträge); `mtp-experts.summary.json` enthält seine Prüfung.

**Ein einzelner Block `mtp.0` ist hier kein vollständiger Ein-Token-Drafter.** Der erste korrekte Meilenstein erzeugt einen Vorschlag über alle drei DSpark-Blöcke und verifiziert einen Draft-Token. Die fünf internen Positionen bleiben zunächst erhalten: Die Draft-Attention ist innerhalb dieses Blocks bidirektional. Abschneiden auf eine Position verändert bereits den ersten Vorschlag. Das wäre ein eigener Genauigkeitsversuch, keine referenzgleiche Implementierung.

## Referenzsemantik

Offizielle Quelle: [DeepSeek inference/model.py, Revision 1bae0e1088993b9b82de08f59de5be5a34e42170](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py). Lokale, unveränderte Kopien und Hashes liegen im Notizenverzeichnis `mtp_referenz/`; Zeilen unten beziehen sich auf diese Datei.

`Transformer.forward` sammelt vor den Blöcken 37/38/39 jeweils den Mittelwert der vier HC-Spuren; ihre Konkatenation hat 15.360 Komponenten (1263–1274). `mtp.0` projiziert sie auf 5.120 und normalisiert **danach**. Dies liefert Attention-Kontext, keine Addition zum Tokenembedding (1131–1139). Die Draft-Eingabe besteht aus dem bereits gezogenen Folgetoken und vier Noise-IDs 128799, eingebettet mit dem Hauptembedding und vierfach repliziert.

`forward_spec` führt `mtp.0 → mtp.1 → mtp.2` mit weitergereichtem HC-Stream und Pre-Mix aus (1278–1285). Die HC-Koeffizienten wirken um eine Sublayer verzögert (972–996). Alle Stufen lesen denselben projizierten Hauptkontext. Ihre Attention verwendet eigene Window-KV, keine komprimierten Hauptmodell-KV (1034–1077). Prefill mit `start_pos=0` füllt nur diese Fenster.

Der letzte Pre-Mix kollabiert die HC-Spuren; `mtp.2.norm` und das geteilte Haupt-Outputgewicht liefern Logits. Der Markov-Head addiert einen vom vorherigen Token abhängigen Vokabularbias. Confidence ist eine separate lineare Ausgabe aus kollabiertem Hidden und Markov-Embedding, **vor** der letzten RMSNorm (1141–1159). Sie steuert in dieser Quelle keinen Akzeptanzalgorithmus. `generate.py` ruft `forward_spec` nicht auf; eine offizielle produktive Speculative-Schleife ist damit nicht belegt.

Das Haupt-Embedding und die Haupt-Ausgabematrix sind zwei verschiedene Gewichte (`tie_word_embeddings=false`); beide werden vom Drafter referenziert, nicht miteinander gebunden. Der Sidecar dupliziert sie nicht.

## Konvertierung und Formatvertrag

Der lokale `engine/convert_hf_to_gguf.py` enthält **keinen** DeepseekV4-/V41-Handler. `skip_mtp` um 8967 gehört zu V2/V3; der Kommentar um 9457 gehört zu Step3.5. Deshalb kann diese Datei den Sidecar nicht zuverlässig exportieren. Tatsächlich relevante Referenzen sind der in `docs/benchmark.md:60` gepinnte [DeepSeek-Konverter](https://github.com/vcruz305/llama.cpp/blob/2076412598e99127ea2554467b2b7530856895d3/conversion/deepseek.py) und [base.py](https://github.com/vcruz305/llama.cpp/blob/2076412598e99127ea2554467b2b7530856895d3/conversion/base.py).

Der neue `engine/scripts/convert_dsv41_mtp.py` übernimmt deren Regeln:

- Experten: FP4-Codes und E8M0-Skalierungsbytes bleiben unverändert; nur die Nibbles werden in `block_mxfp4` umgeordnet. w1 → Gate, w2 → Down, w3 → Up; Experten aufsteigend 0–127 stapeln. Referenz `base.py:743–777`, `deepseek.py:722–749`.
- FP8-Matrizen: E4M3FN nach F32, E8M0-Skalen über **32×32**-Blöcke erweitern, multiplizieren, mit dem lokalen `gguf.quantize(Q8_0)` quantisieren. Referenz `deepseek.py:1246–1274`; Q8_0-Implementierung `engine/gguf-py/gguf/quants.py:380`.
- BF16-Matrizen bleiben BF16, F32-Quellen bleiben F32; Vektoren und Normgewichte werden F32. Keine zusätzliche Transposition/Permutation von `wo_a`; der Graph gruppiert dessen zweidimensionales Layout später. Referenz `deepseek.py:897–909`, allgemeine Normregel; tatsächlich geprüft gegen die Haupt-GGUF.

Alle zehn Hauptdateien wurden mit `gguf-py` inventarisiert: 1.046 Tensoren, 120 MXFP4 / 332 Q8_0 / 65 BF16 / 529 F32. `mtp_referenz/main_tensors.json` hält die Formen/Typen fest. **75 von 81** Sidecar-Tensoren haben ein Hauptlayer-Analogon: gleicher Typ und gleiche Form, mit 128 statt 384 auf der Experten-/Routerachse. Die sechs DSpark-Sondertensoren folgen denselben Quelldatentypregeln. Das ist kein numerischer Inferenzparitätsnachweis.

Alle Formen der folgenden Tabelle sind **ggml-Reihenfolge**, also gegenüber dem Manifest umgekehrt:

| Quelle | Sidecarname | Form / Typ |
|---|---|---|
| `mtp.K.attn.*`, Normen, HC, Router, Shared Experts | `blk.(40+K).*`, exakt die jeweiligen Hauptlayernamen | Hauptform, Router mit 128 Einträgen |
| `mtp.K.ffn.experts.0..127.w1.weight/scale` | `blk.(40+K).ffn_gate_exps.weight` | `[5120,2304,128]` MXFP4 |
| `...w2.weight/scale` | `blk.(40+K).ffn_down_exps.weight` | `[2304,5120,128]` MXFP4 |
| `...w3.weight/scale` | `blk.(40+K).ffn_up_exps.weight` | `[5120,2304,128]` MXFP4 |
| `mtp.0.main_proj.weight/scale` | `blk.40.nextn.main_proj.weight` | `[15360,5120]` Q8_0 |
| `mtp.0.main_norm.weight` | `blk.40.nextn.main_norm.weight` | `[5120]` F32 |
| `mtp.2.norm.weight` | `blk.42.nextn.shared_head_norm.weight` | `[5120]` F32 |
| `mtp.2.markov_head.embed.weight` | `blk.42.nextn.markov_embd.weight` | `[256,129280]` BF16 |
| `mtp.2.markov_head.head.weight` | `blk.42.nextn.markov_output.weight` | `[256,129280]` BF16 |
| `mtp.2.confidence_head.proj.weight` | `blk.42.nextn.confidence_proj.weight` | `[5376,1]` BF16 |

Die Attentionnamen sind u.a. `attn_sinks`, `attn_kv_a_norm`, `attn_q_a_norm`, `attn_kv`, `attn_q_a/b`, `attn_output_a/b`. Routingbias bleibt `exp_probs_b.bias`, Visionbias `exp_probs_b_vl.bias`; letzterer wird erhalten, beim Text-Drafting nicht benutzt. HC-Tensoren bleiben vollständig F32. Kein Quelltensor wird still verworfen: 2.304 Experten-Quelltensoren werden 9 Stapel, 25 FP8-Skalentensoren gehen in die Q8_0-Konversion ein, 72 weitere Quellen werden einzelne Tensoren: insgesamt 81.

Metadaten: `general.architecture=deepseek41`, `general.type=mtp-sidecar`, `deepseek41.mtp.format=inferssd-dspark-v1`, Basislayer 40, drei Stufen und komplette Originalkonfiguration als JSON. Die `nextn.*`-Sondernamen sind **neuer Sidecarvertrag**, keine Behauptung bereits vorhandener Loader-Unterstützung. Das File ist kein normales `--model-draft`-Modell; Tokenizer und Hauptgewichte fehlen absichtlich. Den Hauptmodellwert `n_layer=40` niemals global auf 43 erhöhen.

Reproduktion im Worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 ~/work/mtp-venv/bin/python engine/scripts/convert_dsv41_mtp.py --self-test
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  nice -n 19 ~/work/mtp-venv/bin/python engine/scripts/convert_dsv41_mtp.py
PYTHONDONTWRITEBYTECODE=1 nice -n 19 ~/work/mtp-venv/bin/python \
  engine/scripts/convert_dsv41_mtp.py --verify-only
```

Konversion verweigert vorhandene Zieldateien und unvollständige Shards; erst nach Rückleseprüfung wird `.gguf.partial` umbenannt. Bestehenden Sidecar mit `--verify-only` prüfen oder `--output` auf einen neuen Pfad setzen. Venv-Pakete: NumPy, tqdm, PyYAML sowie safetensors; letzteres wurde installiert, ist dank Wiederverwendung von `gguf.utility.SafetensorsLocal` nicht erforderlich. Kein Torch. Die drei Shards sind headerseitig vollständig (2.652.728.736 / 2.573.998.176 / 2.706.402.896 Byte); `mtp-laden`: inactive, Result success, ExecMainStatus 0.

Gemessene Konversion inklusive vollständiger SHA256-Rückleseprüfung: 33,81 s, Exit 0, keine Swaps dieses Prozesses. MaxRSS 7.562.204 KiB einschließlich sauberer mmap-Seiten beim Verifizieren; maximal ein Expert-Stapel von ca. 765 MiB wird als Ausgabe aufgebaut. Das ist eine Dateikonversionsmessung, keine Decode-Messung.

## Loader und SSD-Expert-Tier

Vorgeschlagener Einstieg: **`--mtp-sidecar PATH`**, standardmäßig leer. Er führt erst bei expliziter Angabe zu zusätzlichen Gewichten, Featureausgängen oder Draft-Arbeit. Der bestehende separate Draft-Modellpfad darf den Sidecar nicht als vollständiges Modell laden.

1. Ein vom Hauptmodell besessenes DSpark-Gewichtsobjekt hält einen eigenen GGUF-Reader, Tensor-Kontext und die drei Layer. Reader/Mappings/Dateihandles müssen alle Graphen und SSD-Zugriffe überleben. Gemeinsame Tensoren verweisen auf `tok_embd` und `output`; es gibt keine zweite Kopie und keine zweite Vokabularinitialisierung.
2. Vor Veröffentlichung Formatversion, alle 81 Namen/Formen/Typen und Konfiguration prüfen: hidden 5120, vocab 129280, HC 4, 3 Stufen, Zielindizes 37/38/39, Block 5, Markov 256, MoE 128/top-3, Shared 1, Skalierung 1,5, SwiGLU-Limit 10, Norm-Epsilon 1e-20. Falsches Hauptcheckpoint/Tokenizer als Fehler behandeln; reine Modellnamensgleichheit ist kein belastbarer Identitätsnachweis. Quellenrevision/Hashes im Manifest zur Bindung ergänzen, sobald ein Loader diese prüft.
3. Hauptlayer-Muster für normale Tensorerzeugung steht in `engine/src/llama-model.cpp:5999`; Sondergewichte separat erzeugen. `wo_a` bleibt 2D wie bei `:6034`. Dense Gewichte über bestehende Backend-Pufferregeln; geroutete MXFP4-Experten über denselben Deferred-/CPU-SSD-Pfad (`:3568`). Die Referenz hebt Confidence-Projektion und Head-Akkumulation zur Laufzeit auf F32 an; die BF16-Speicherung wird dadurch nicht geändert.
4. `blk.40..42` sind globale SSD-Namens-/Layer-IDs. Der Drafter erhält lokale Stufen 0..2 und explizite globale IDs 40..42. Haupt-KV-/Attention-Schleifen bleiben 0..39. Speicher-/Tier-Registrierung muss die drei zusätzlichen IDs hingegen kennen. Nicht blind `describe_experts(..., 40)` aufrufen: es liest `swiglu_clamp_exp[layer]` (`engine/src/ssd/model-adapter.cpp:11`); Clamp und Top-k müssen aus der Draft-Konfiguration kommen.
5. Die aktuelle Context-Registrierung scannt nur `model.hparams.n_layer` und füllt `expert_shapes` aus Hauptlayern (`engine/src/llama-context.cpp:311–328`). Hier zusätzliche MTP-Expertformen registrieren: je Projektion `nb[2]=6.266.880` Byte, `ne[2]=128`; Tier-Prioritätsliste bis Layer-ID 42 zulassen. `llama_kimi_cpu_sink_runtime` akzeptiert eine Layerzahl separat, filtert aber IDs außerhalb dieser Zahl (`engine/src/ssd/cpu-runtime.h:97–114`). Globale 384 Experten sind kein Ersatz für die tatsächliche tensorweise Zahl 128. Cache-/VRAM-Budget bleibt gemeinsam begrenzt; keine zweite unkoordinierte VRAM-Tier-Instanz.

Der vorhandene Indexer erzeugt den Sidecarindex bereits ohne Änderungen:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=framework/src nice -n 19 \
  ~/work/mtp-venv/bin/python -m inferssd.source.gguf_alias \
  --model-glob /daten/inferssd-models/DeepSeek-V4.1-Flash-1017-MTP/DeepSeek-V4.1-Flash-1017-MTP-Q8_0.gguf \
  --n-experts 128 \
  --out /daten/inferssd-models/DeepSeek-V4.1-Flash-1017-MTP/mtp-experts.tsv \
  --summary-json /daten/inferssd-models/DeepSeek-V4.1-Flash-1017-MTP/mtp-experts.summary.json
```

`expert.source` bekommt den bestehenden Hauptindex **und zusätzlich diesen TSV-Pfad**, nicht den GGUF-Pfad. `engine/ggml/src/ggml-ssd/store/index.cpp:19–39` unterstützt mehrere Indizes und lehnt direkte GGUF-Pfade ab. TSV-Schlüssel: Tensorname, Expertenindex, logische Bytezahl; Offset ist die tatsächliche GGUF-Payloadposition plus `expert_idx × 6.266.880` (`store/parser.cpp:235–305`). Es gibt 384 MTP-Experten und 1.152 Projektionseinträge. Separat indexieren: der Python-Indexer verlangt innerhalb eines Aufrufs eine einheitliche Expertenzahl und einen vollständigen GGUF-Shardsatz (`framework/src/inferssd/source/gguf_alias.py:327–418`); Hauptdateien und Sidecar lassen sich nicht einfach in denselben Aufruf stecken.

Die aktuelle GGUF-Ausrichtung ist 32 Byte; alle 1.152 Einträge sind laut Indexer gegenüber 4096 Byte unausgerichtet. Das ist kein Code-/Skalenfehler. Der native I/O-Pfad muss die bestehenden erweiterten, ausgerichteten Reads verwenden und die logischen Offsets erhalten; nicht TSV-Offsets herunterrunden oder `--require-aligned` vortäuschen. Ein zusätzlicher CPU-only-Check mit dem echten `source_index` hat alle 1.152 Extents aufgelöst, pro Tensor 128 Blöcke erkannt, Expertenindex 128 zurückgewiesen und je Extent Anfang/Ende mittels ausgerichtetem `O_DIRECT`-Read gegen gepufferte Dateibytes geprüft: **2.304 Reads bestanden**. Quelle/Binary: `mtp_referenz/check_mtp_index.cpp` / `check_mtp_index`; Protokoll `native-index-verification.log` beim Sidecar. Ein integrierter Model-/Tier-Loader-Test bleibt offen.

## Compute-Graph: ein verifizierter Draft-Token

Die nachfolgende Umsetzungsskizze leitet sich aus der Referenz ab; sie ist noch nicht implementiert.

**Feature-Ausleitung.** Vor dem bestehenden HC-Mix der Hauptlayer 37/38/39 (`engine/src/models/deepseek41.cpp:1147–1166`) jeweils Mittelwert über die HC-Achse bilden und in dieser Reihenfolge konkatenieren. Keine letzte Output-Norm verwenden. Für akzeptierte Positionen müssen diese Features auch bei `batch.logits=false` verfügbar sein, insbesondere beim Replay. Auf dem Backend halten und nur nötige letzte 128 Positionen pro Sequenz puffern.

Die vorhandene `embeddings_nextn`-API reicht dafür unverändert nicht: ihr Puffer ist auf `n_embd` pro Token ausgelegt (`engine/src/llama-context.cpp:2529`, `:2642`), DSpark braucht `3*n_embd`. Außerdem setzt der V4.1-Graph `t_h_nextn` derzeit gar nicht; sein Ende enthält nur finale Embeddings und Logits (`models/deepseek41.cpp:1247–1259`). Die Ergebnisstruktur um einen eigenen DSpark-Featuretensor/Getter mit expliziter Breite erweitern oder den vorhandenen Transport konsistent um diese Breite ergänzen. Kein 15.360-Float-Tensor in einen 5.120-Float-Puffer schreiben.

**Context und Positionen.** `main_proj` und `main_norm` erzeugen pro Hauptposition `main_x`. Jeder Draftlayer projiziert dieses mit seinem eigenen `wkv`, KV-Norm und RoPE und hält einen eigenen Ring mit 128 Zeilen. Kontext liegt bis einschließlich der letzten bereits berechneten Hauptposition vor. Prompt-Prefill befüllt die drei Ringe. Als einfacher Start wird anschließend ein gewöhnlicher Hauptdecode-Schritt ausgeführt, bevor Drafting aktiviert wird; dadurch ist der `start_pos=0`-Sonderfall ohne neue Semantik abgedeckt.

**Drafting.** Fünf Tokenpositionen: bekanntes, bereits vom Hauptsampler gezogenes Token plus vier Noise-Tokens. Der initiale HC-Pre-Mix wählt Spur 0. `main_x` geht ausschließlich in den Kontextzweig. Draft-Q/KV bekommen die absoluten Positionen hinter dem Hauptkontext. Attention liest die gültigen Hauptkontext-Ringzeilen und alle fünf temporären Draft-KV; diese fünf KV werden nicht als akzeptierter Kontext persistiert. Attention-Sink und inverse Output-RoPE wie im Hauptgraph verwenden, aber nicht dessen kausale Draftmaske oder Compressor/Indexer übernehmen.

**Drei Blöcke.** Vorhandene HC-/RMSNorm-/Attention-Projektions-/Shared-Expert-/MoE-Operationen wiederverwenden. Die verzögerten Pre-Mixes wie in `models/deepseek41.cpp:1159–1188` erhalten. MoE explizit 128 Experten, Top-3, Textbias, `sqrtsoftplus`, normalisierte Gewichte und Scale 1,5. Für die fünf Positionen den bestehenden gruppierten CPU-Sink-Prefill aktivieren: `engine/src/llama-graph.cpp:2046`, Adapterprüfung `engine/src/ssd/model-adapter.cpp:57`. Kein neues Expert-Compute-System. Der Hauptverifier benötigt denselben Mehrtoken-Sink-Pfad für zwei Eingaben.

**Output.** Nach Block 42 über den letzten Pre-Mix kollabieren, Draft-Outputnorm und gemeinsame Ausgabematrix anwenden. Für Vorschlag 1 den Markov-Bias des bekannten Tokens addieren; nur diesen Vorschlag zurückgeben. Die übrigen vier Outputprojektionen/Sampling-Schritte und Confidence-Auswertung können entfallen, sobald ihre Unabhängigkeit vom ersten Output anhand des Graphen bestätigt ist. Ihre Hidden-Zustände innerhalb der drei Blöcke bleiben nötig. Einen eigenen Draft-Sampler/RNG verwenden; anfänglich genügt deterministisches Argmax. Der Ziel-Sampler darf beim Drafting weder RNG, Grammar noch Repetition-State verändern.

## Server-Verifikation und exakte Ausgabe

Neue DSpark-Implementierung an `common_speculative_state::{begin,process,draft,accept}` anbinden (`engine/common/speculative.cpp:117–166`), vorhandene Batch- und Samplerlogik wiederverwenden. `process` darf unverifizierte Hauptfeatures zunächst nur vormerken; das Commit erfolgt für den tatsächlich akzeptierten Präfix. Der bereits vorhandene EAGLE3-Eintrag ist keine DSpark-Implementierung.

Sei `P` der berechnete Zielpräfix, `u` das schon gezogene/ausgegebene, aber noch nicht decodierte Token an Position `p`. DSpark schlägt `v` für `p+1` vor:

1. Checkpoint von `P` vor der Verifikation anlegen; auch letzte Hauptfeatures und Draft-Ringe/Positionen konsistent sichern.
2. Eine Zielbatch `[u, v]` an Positionen `[p, p+1]` mit beiden Logitausgängen decodieren. Die vorhandene Batchbildung macht genau dieses Muster (`engine/tools/server/server-context.cpp:445–453`). Nicht als zwei unabhängige Sequenzen darstellen. Für beide Tokens alle HC-/Compressor-/Indexer-Abhängigkeiten innerhalb der Batch erhalten.
3. `common_sampler_sample_and_accept_n` (`engine/common/sampling.cpp:676–703`) zieht aus Logitzeile 0 das Zieltoken `v*`. Bei `v*=v` ist der Draft akzeptiert; dann aus Zeile 1 das Bonus-Token `w*` ziehen. Sonst nach `v*` stoppen. Nur tatsächlich ausgegebene Zieltoken in Grammar/Penalties/RNG akzeptieren. Das bekannte `u` nicht erneut sampeln.
4. Bei Ablehnung Zielzustand zu `P` zurücksetzen, nur `u` erneut decodieren. `v*` bleibt das nächste noch nicht decodierte Token. Bei Akzeptanz dürfen nach erfolgreichem Cache-/Paritätsnachweis die beiden berechneten Eingaben bestehen bleiben. Accepted-Features aller committed Eingaben in die drei DSpark-Ringe übernehmen; bei zwei Eingaben genügt nicht bloß das letzte Feature.

Dieser Vergleich mit echten Ziel-Samples ist für jede Temperatur korrekt und erhält bei identischen Ziel-Logits und identischem RNG-Verbrauch dieselbe Tokenfolge wie der sequentielle Sampler. Er benötigt keine Draft-Wahrscheinlichkeitsquotienten und kein Residualsampling. Standard-Speculative-Acceptance über `min(1,p/q)` garantiert dagegen im Allgemeinen nur dieselbe Verteilung, nicht dieselbe Folge zum gleichen Seed.

**Offene harte Bedingung:** Zwei-Token-Matmul/Attention/CPU-Prefill und zeitabhängige CPU/GPU-Expert-Platzierung können andere Rundung erzeugen als Ein-Token-Decode. Der heutige Server benennt die Semantik ausdrücklich `batch-verifier` (`server-context.cpp:3544–3551`); Replay macht bereits gezogene Batch-Samples nicht rückwirkend sequentiell identisch. Vor Freigabe bitgleiche per-position Logits nachweisen oder die Abweichung im gemeinsamen Kernelpfad beheben. Wenn dies nicht gelingt, muss ein strikt identischer Modus sequentiell verifizieren bzw. vor RNG-Verbrauch auf den normalen Pfad zurückfallen. Dann ist kein Speculative-Speedup zugesichert. Das darf nicht als bestandener Äquivalenztest ausgegeben werden.

## Rollback: einfach korrekt anfangen

### Was heute fehlt

- `n_rs_seq>0` wird explizit abgelehnt, weil `split_equal` keinen sicheren Tail-Erhalt unterstützt (`engine/src/llama-kv-cache-dsv4.cpp:947–952`, `:1228`). Den Guard zu entfernen aktiviert keinen korrekten Rollback.
- `seq_rm(p0,-1)` innerhalb berechneter Tokens scheitert bei `n_rs_seq=0` (`:1280–1326`). Nur Raw-KV-Metadaten zu löschen stellt die Compressor-Zwischenzustände nicht wieder her.
- Die vorhandenen Snapshot-/Restore-Graphhelfer (`engine/src/models/deepseek41.cpp:74–126`) und Planes (`llama-kv-cache-dsv4.cpp:426–485`) sind ohne den unterstützten Splitter kein fertiger Lösungsweg.
- Der Zweig für `p0>pos_max` entfernt komprimierte Metadaten mit **festen 4/128** (`llama-kv-cache-dsv4.cpp:1293–1297`). V4.1 ermittelt seine Ratios dynamisch als **2 und 1** (`:995–1030`). Ein Restore mit anschließendem blindem `seq_rm` kann damit falsche Grenzen anwenden. Für DSpark diesen Zweig vermeiden; eine spätere gemeinsame Korrektur muss tatsächliche Cache-Ratios verwenden und ratioübergreifend getestet werden.
- **Auch der vollständige Serializer braucht vorher eine Korrektur:** `state-dsv4.cpp:403–404` speichert HCA-Zeilen mit dem festen Divisor 128. Der V4.1-HCA-Speicher und sein Zustand werden aber mit `ratio_b=1` erzeugt (`llama-kv-cache-dsv4.cpp:1114–1119`, `:1138–1140`). Dadurch fehlen bei einem Sequenz-Snapshot viele gültige 1:1-Zeilen. Vor Nutzung des vollständigen Rollbacks hier `hca_state->get_ratio()` verwenden und einen Roundtrip-Test für V4.1 und V4/128 ergänzen. CSA/LID werden heute mit `ratio_kv=1` ausreichend breit gesichert; ein konsistenter späterer Umbau kann auch dort die tatsächliche Zustandsratio verwenden. Dieser Befund ist codebasiert, nicht durch einen Modelllauf reproduziert.

### Meilenstein A: vollständiger bestehender Sequenz-Snapshot

Nach der HCA-Zeilenkorrektur `llama_state_seq_get_*_ext(..., flags=0)` vor der Verifierbatch verwenden und bei Reject mit denselben Flags zurückspielen. Das vollständige DSv4-Format umfasst Raw/SWA-Zellen mit Token-ID/Position/Sequenz, komprimierte K-Zeilen und alle drei Compressor-KV-/Score-Zustände: `engine/src/inferssd/state-dsv4.cpp:372–415`. Restore löscht komprimierte Speicher und lädt den gesicherten Zustand (`:418–458`); der Raw-Restore ersetzt die Sequenz (`engine/src/llama-kv-cache.cpp:2198–2245`). Ergebnis muss exakt am vorherigen Prefix-Ende liegen. **Kein anschließendes generisches DSv4-Suffix-`seq_rm`** für das bereits restaurierte Präfix. Unverändert ist der heutige vollständige Sequenz-Snapshot wegen des Divisors 128 für diesen Einsatz nicht korrekt.

Nach Restore nur die tatsächlich committed Eingaben erneut ausführen: Reject `[u]`, Accept gegebenenfalls `[u,v]`. Das zuletzt gezogene Token bleibt undecodiert. Engram-Historie hängt an `ext.tok` der Raw-Zellen (`models/deepseek41.cpp:192–218`), deshalb diese IDs und die Server-Tokenliste mitrestaurieren. Samplerzustand **nicht** auf vor die gerade gültig gezogenen Zieltoken zurücksetzen; ein gesonderter Sampler-Checkpoint wird nur für einen Abbruch vor Veröffentlichung bzw. eine erneute sequentielle Entscheidung benötigt. Pro Sequenz getrennte MTP-Ringe und Feature-Queues sichern/zurücksetzen. Nach Restore Graphplanung/Eingaben aus dem restaurierten Zustand neu erzeugen, keine bereits geplanten Kompressions-Indizes wiederverwenden.

Für die erste Korrektheitsprüfung kann man nach **jedem** Verifierblock restaurieren und die committed Eingaben sequentiell replayen. Das entspricht dem Grundmuster von `server-context.cpp:2173–2245`, aber nicht dessen ungeprüften DSv4-Details. Der aktuelle generische Pfad benutzt `PARTIAL_ONLY` (`:2181`) und danach `seq_rm` (`:2191`), außerdem replayt er auch nach vollständiger Akzeptanz (`:3522–3529`). Für den ersten DSv4-Meilenstein getrennt auf vollständigen Snapshot und sichere Restore-Grenzen umstellen. Vorhandener Code ist hier kein Nachweis fertiger DSv4-Spekulation.

Vorteil: keine neue Cache-Architektur, keine Annahme über wiederherstellbare Partial-Blöcke. Preis: der vollständige Snapshot kopiert komprimierte Historie proportional zur Kontextlänge und kann den gesamten Speedup aufzehren. Dies ist der Korrektheitsmodus, nicht die endgültige Performancebehauptung.

### Meilenstein B: begrenzter Snapshot der betroffenen Zeilen

Erst nach bestandenen Tests den vorhandenen `PARTIAL_ONLY`-Serializer nutzen: Er sichert Raw/SWA und die kleinen Compressor-Zustände, lässt komprimierte K-Zeilen aber aus (`state-dsv4.cpp:373`, `:401–419`). Für zwei neue Eingaben werden geschlossene 2:1-Zeilen und 1:1-Zeilen angehängt. Bestehende Zeilen bleiben bei gültigem Append-Vertrag unverändert. Dies für alle KV-/Indexquellen und Dummy-Schreibziele beweisen; kein pauschales „Fenster zurücksetzen reicht“.

Kleinster belastbarer Zusatz ist ein Undo-Snapshot der **tatsächlich vom Zwei-Token-Plan beschriebenen Zielzeilen** plus Metadaten/Heads: Raw-Ring, CSA-/HCA-K und Lightning-Indexer-K; dazu die existierenden KV-/Score-Ringzustände, `rs_idx` und MTP-Ringe. Ziele aus `dsv4_build_comp_plan` erfassen (`llama-kv-cache-dsv4.cpp:197–407`), nicht über festes `p/4` schätzen. Auch Padding-/Dummyzeilen und zusammenhängende Backend-Abhängigkeiten berücksichtigen. Bei Reject restore, verworfene Append-Metadaten mit den **jeweiligen** Ratios beschneiden, `u` erneut ausführen. Alles synchronisieren, bevor ein anderer Slot dieselben Speicherstrukturen verändert; keine prozessweiten kompletten Restore-Snapshots auf parallele Slots anwenden.

Reines erneutes Ausführen der abgelehnten Position ohne Restore ist nicht hinreichend: Der 128er Raw-Ring kann alte Zeilen überschrieben haben und die Compressor-/Indexer-Ringe enthalten schon die falsche Position. Candidate-/Top-k-Tensoren sind im aktuellen Graphen abgeleitete Zwischenwerte (`models/deepseek41.cpp:921–1110`), kein separat langlebiger Cache; nach Restore neu berechnen. Falls deren Lebenszeit später über Graphaufrufe verlängert wird, gehören sie ebenfalls in den Transaktionszustand.

Alternativ kann später der Upstream-Tail-Splitter mit seinen Rollback-Planes portiert werden. Für genau einen Draft ist dieser große Port nicht der erste Schritt. `state_write_at` hilft ebenfalls nicht allgemein: frühere Snapshots sind an gemeinsame Kompressionsgrenzen gebunden (`state-dsv4.cpp:328–369`); einen bereits abgelehnten beliebigen Zustand kann man damit nicht nachträglich rekonstruieren.

## Testplan und Freigabekriterien

Bereits erledigt: `--self-test` für Nibbles, FP8-Subnormalwerte, 32×32-Skalen in beiden Dimensionen, Chunk-Grenze 256, BF16-Biterhalt, F32-Normen und Byteformen; vollständiger Konversionslauf; 75 Analogonvergleiche; 81 Rücklese-Hashes; strenger Aliasindex-Headercheck und nativer `source_index`-/Direct-I/O-Check aller Einträge. Keine Inferenzäquivalenz behauptet.

Als nächste Tests, zunächst kleine künstliche Fixtures CPU-only, reale Modelltests erst außerhalb des Produktionsbetriebs:

1. **Loader/SSD:** fehlender/abgeschnittener Sidecar, falsche Dimension/Revision, Namenskollision und falscher Typ müssen vor Context-Publikation scheitern. Off-Flag darf keine zusätzlichen Ressourcen allozieren. Alle 1.152 Einträge über den nativen `source_index` auflösen und ausgerichtete Direct-Reads mit logischer GGUF-Bytefolge vergleichen. 128 als echte Grenze testen (127 gültig, 128 ungültig); Hauptlayer mit 384 weiterhin gültig. Tier-Eviction/Prefetch zwischen Haupt- und Draftlayern ohne globale Budgetüberschreitung.
2. **Graph:** gespeicherte kleine Referenz-Zwischentensoren gegen Engine vergleichen: Eingangs-HC-Mittel, Konkatenationsreihenfolge, Projektion/Norm, Ringrotation, fünf bidirektionale Draftpositionen, verzögerte HC-Pre-Mixes, Top-3-Router, gemeinsamer Head plus Markov-Bias. Keine PyTorch-Inferenz im Produktionsprozess. Draft-Quantisierung gegen entsprechend quantisierte Referenz beurteilen, nicht gegen unquantisierte Logits bitgleich verlangen.
3. **Cache:** Zwei-Token-Schritt mit erzwungenem Reject, Restore und Replay `[u]` muss denselben logischen Cache samt Folgelogits liefern wie sequentiell `[u]`. Akzeptanz und Bonus analog `[u,v]`. Positionen um 2er-Kompressionsgrenzen, 127/128/129, gepolsterte 256er-Breiten und lange Index-/Candidate-Top-k-Übergänge abdecken. Reste offener Compressor-Blöcke, KV-/Indexquellen, `ext.tok`, Slotwechsel und mehrere Sequenzen prüfen. Bytevergleich gespeicherter **gültiger** Zustände plus Folgedecode; tote Paddingzeilen gesondert bewerten.
4. **Greedy:** feste Prompts und gleiche Flags/Expert-Platzierung; Baseline ohne MTP gegen MTP mit Draft 1. Beide Verifier-Logitzeilen mit sequentieller Referenz vergleichen, dann sämtliche Token-IDs bis EOS/Stop. Reject und Accept gezielt erzwingen; keine bloße Textgleichheit.
5. **Temperatur > 0:** identischer Seed und vollständige Samplerkette (top-k/p, min-p, Penalties, Grammar/Tool-Constraints). Ziel-RNG nur pro tatsächlichem Outputtoken verbrauchen; Draft-RNG unabhängig. Über viele Seeds tokenweise Gleichheit verlangen, auch bei frühem Reject und Bonus-Token. Logits-Bitparität ist Voraussetzung für die strenge Zusage. Eine ähnliche Verteilung oder gleiche greedy-Ausgabe ersetzt diesen Test nicht.
6. **Fehlerpfade:** fehlgeschlagene Batch/Restore, Abbruch/Stop/EOS zwischen Outputs, Contextgrenze, Promptcache-Reuse und Prefill-Chunks. Keine kontaminierten Features oder Draft-Ringe im Folgerequest. OOM-Verhalten aus Ergebnis G bleibt unabhängig zu beachten; kleine Batchgröße beseitigt den direkten CUDA-Allokationsfehler nicht prinzipiell.

## Erwartete Beschleunigung

Für einen Vorschlag liegt die ideale Grenze unter 2×. Sei `a` dessen Akzeptanzrate, `T1` sequentielle Haupttokenzeit, `T2` Zeit der gemeinsamen Zwei-Token-Verifikation, `D` die drei DSpark-Blöcke mit fünf internen Positionen, `C` Snapshot-/Restore-Aufwand und `R` Replay eines Rejects. Ein späterer Modus ohne Replay bei voller Akzeptanz erreicht näherungsweise:

```text
Speedup = (1 + a) * T1 / (T2 + D + C + (1 - a) * R)
```

Bei dem aus dem Branchkontext bekannten Größenordnungswert `T1≈116 ms` (8,6 tok/s), **angenommenen**, nicht gemessenen `a=0,8`, `T2=145 ms`, `D=15 ms`, `C=5 ms`, `R=116 ms` ergibt das etwa **1,11×**. Bereits `T2=180 ms`, `D=30 ms`, `C=20 ms` ergäben nur **0,82×**. Keine Akzeptanzrate oder Draftzeit dieses Sidecars wurde gemessen. Ein vernünftiges erstes Performanceziel nach Cacheoptimierung ist 1,0–1,3×; auch ein Verlust ist möglich. Das gemeinsam gelesene Expertgewicht ist der erhoffte Gewinn, die fünf Draftpositionen und zusätzlicher Tier-Druck sind reale Kosten.

Der Korrektheitsmodus mit Replay nach jedem Block addiert ungefähr `(1+a)*T1` zum Nenner und ist erwartbar langsamer. Vollständige Kontext-Snapshots können ebenfalls dominieren. Erst getrennte Zeiten für Draft, Verifier, Snapshot/Restore, Replay, Akzeptanz, SSD-Bytes und Tier-Misses messen; nicht fremde DSpark-Speedups oder frühere CPU-Sink-Zeitanteile übernehmen.

## Nächste Implementierungsschritte

1. HCA-Snapshotzeilenzahl korrigieren und einen DSv4-Zustands-Roundtrip für die tatsächlichen Ratios testen; optionalen Sidecarloader mit Format-/Formprüfung und gemeinsamem Expert-Index/Tier implementieren. Gegen CPU-Fixture prüfen und mit `JOBS=8 bash ~/work/inferssd-opt-mtp-build.sh` bauen.
2. Feature-Ausleitung für 37/38/39 und drei DSpark-Blöcke, zunächst Draft 1, voller interner Block 5.
3. Vollständigen DSv4-Restore ohne den falschen Suffix-Trim integrieren; Cache- und Samplingäquivalenz nachweisen.
4. Erst danach Partial-/Zeilen-Snapshots, Akzeptanz-Fastpath und Performanceversuche. Weitere Drafttokens/Confidence-Steuerung bleiben spätere Schritte.

In diesem Arbeitsstand wurde kein C++-Loader begonnen und kein Engine-Build ausgeführt; nur der kleine native Indexcheck wurde separat mit g++ gebaut. Die kritische Semantik und die Rollback-Grenzen sind vor einer solchen Änderung geklärt; Server-/GPU-Laufzeittests bleiben offen.
