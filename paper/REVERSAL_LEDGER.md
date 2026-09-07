# Reversal Ledger — the audit chain at a glance

Every surviving number in this paper has an alibi. Nine times, a claimed result was
overturned by a harness or protocol defect; each overturning followed the same loop —
**discovery → forensics → repair → withdrawal → re-measurement → assertion pin** —
and each left a machine-checkable anchor behind. This table is the citable form of
that chain (compact version for the appendix / tech report front matter).

| # | Withdrawn claim | Root cause | How it surfaced | Repair | Assertion anchor |
|---|---|---|---|---|---|
| 1 | Near-field "+26.5% @w=128" (original submission) | a crashed NIAH run's numbers were mislabeled as results | provenance audit: table values absent from run logs | honest withdrawal; ws=256 re-run | FORBIDDEN list; CLAUDE.md |
| 2 | "Near-field *degradation*" (v2 protocol) | loss computed on the query phrase's own continuation — answer never in the input | code review reading the loss slice | answer appended; loss on answer region (A.4) | v4/v5 protocol; per-region loss test |
| 3 | Near-field "+20.6–40.2%" (v4) | second haystack truncation silently cut the needle tail — near bucket seeded with unanswerable samples | package review + geometric re-check | truncation before insertion; per-example + aggregate survival asserts (A.9) | A12: six v5 anchors, survival 1.0 |
| 4 | "Multiscale has no independent contribution" | over-generalization without the gate-off control | gate-off cells in the 2×2×2 grid | reworded "real but secondary (+0.7–3.0pp)" | Table 2 gate-off cells; A2b |
| 5 | "M4L8 is unstable" | NaN *gradient infection* misread as method instability | guard-isolation experiment | total-norm isfinite guard + NaN-skip; trains to 25.7% | A3; ladder row |
| 6 | "decode = local (116 tok/s)" | patched forward never wrote the KV cache → benchmark measured 1-token self-attention | package review (P0-1) | true sliding-window KV cache; paired re-measurement (A.8) | A10: six paired measurements, median 1.004 |
| 7 | LongBench "generation collapse" (local ≈0.0001) | same decode defect — generate() sliced input to the last token | before/after asymmetry (0.0001→0.0332 with cached≡recompute test-pinned) | re-run under fixed path; full arm reproduces exactly | A14: closure ≤1.3%, full 0.0804 |
| 8 | "LoRA 87.6% / dwconv 78.3%" | adapter wrappers never wired into the forward path (zero-init ⇒ silent no-op ≡ local) | canonical first run: 16-digit match with the local baseline | wiring + positive-control assert (A.6) | A11: 87.4% / 78.3%, load_check clean |
| 9 | "Prefill 5–8× (M1L8)" | systems loader default measured the M4-gate-on structure | explicit M1L16 measured *faster* than "smaller" M1L8 — structurally impossible | explicit structure flags; same-session pairs (A.7) | A13: 2.9×/4.8×/8.7× scaling with units |

Supporting hygiene fixes (same loop, smaller blast radius): γ-only@M1L8 9.5→10.0
(in-loop vs canonical caliber); Table 3 chimera (old NLLs + v4 percentages) rebuilt
from one file; 6-task vs 7-task means unified; decode OOM from heap fragmentation
reordered; triviaqa padded-len buffer accumulation bounded in-module; per-sample
spectral-cache hygiene in the LongBench harness.

**Freeze declaration.** The experiment matrix is frozen at commit `1bff7cdc`
(+) the text-level close-out commits; `paper/recompute_tables.py` — 30+ assertions
covering every table, the attribution ladder, NIAH v5, LongBench, systems, gate
selectivity, and the official-test replication — passes **ALL** at the freeze point.
No further experiments are planned or permitted without a pre-registered protocol
and a named claim they gate.
