# Revision Data Audit — Phase 0 (2026-09-05)

Recomputed every table number from `results/*.json`. Sources are absolute; any old value
that could not be reproduced from a results file is marked UNTRACEABLE and replaced.

## Table 1 (PG-19 main results)

- GPT2-XL w=128 train@512: n=5 seeds, local=2863.9±476.2, full=33.7, MSRA=2751.2±455.9, gain=3.9%, gap_closed=4.0%  → paper 2,864→2,751 +3.9% **VERIFIED** (paper said '5-seed', correct)
- LLaMA w=128 train@512 (final config M=1 gate-off L=8): n=3 seeds (final_M1_L8_s2-4), local=355.6±156.6, full=17.0, MSRA=312.3±134.9, gain=12.2%, gap_closed=12.8%  → paper '336→301 +10.3% (5-seed)' **UNTRACEABLE — REPLACE**
  (alt: M=4 gate-on config llama_512_s1-3: n=3, local=327.8, MSRA=313.9, gain=4.3%, gap_closed=4.5%)
- LLaMA w=128 train@1024: n=4 seeds (t1024_ws128_s2-5), local=510.4±91.0, full=17.2, MSRA=474.5±84.0, gain=7.0%, gap_closed=7.3%  → paper '521→479 +8.0% (5-seed)' **WRONG: +8.0%→7.0%, 5→4 seeds**
- LLaMA w=256 train@1024: n=4 seeds (t1024_ws256_s2-5), local=223.0±48.4, full=17.2, MSRA=207.4±44.3, gain=7.0%, gap_closed=7.5%  → paper '227→208 +8.2% (5-seed)' **WRONG: +8.2%→7.0%, 5→4 seeds**

## Table 2 (window sweep)

- w=64 (llama_ws64.json, M=4 config): local=523.4, full=12.5, MSRA=503.0, gain=3.9%, gap_closed=4.0%  → paper 523.4/503.0 +4.0% **VERIFIED**
- w=128: use Table 1 final_M1_L8 series → local=355.6, full=17.0, MSRA=312.3, gap_closed=12.8%  → paper '335.5/300.9 +10.3%' **UNTRACEABLE — REPLACE**
- w=256 (llama_ws256_retry.json, M=4 config): local=74.1, full=13.4, MSRA=69.9, gain=5.6%, gap_closed=6.9%  → paper 74.1/69.9 +6.9% **VERIFIED**
  (Reviewer ve2u's 5.7% is the relative-PPL-reduction reading; 6.9% is correct under the gap-closed definition — fix is to ADD the definition + Full PPL column, not change the number.)
  NOTE: w=64 and w=256 rows use the M=4 gate-on config; w=128 row (rebuilt) uses M=1 gate-off — caption must state per-row configs.

## Table 4 (LongBench, perplexity-space gap closed, ws=128 columns)

- gov_report: full=5.5, local=479.0, msra=352.5, gap_closed=26.7%  [OK]
- qasper: full=10.1, local=1255.6, msra=2595.0, gap_closed=-107.5%  [OK]
- passage_retrieval_en: full=84.4, local=1534.9, msra=1693.6, gap_closed=-10.9%  [OK]
- narrativeqa: full=24.2, local=3169.7, msra=1441.4, gap_closed=54.9%  [OK]
- qmsum: full=21.4, local=458.8, msra=372.6, gap_closed=19.7%  [OK]
- multi_news: full=7.4, local=1160.1, msra=924.6, gap_closed=20.4%  [OK]
- triviaqa: full=16.9, local=9234.7, msra=1145.0, gap_closed=87.8%  [OK]
- hotpotqa: full=11.1, local=3370.6, msra=4020.0, gap_closed=-19.3%  [OK]
- 2wikimqa: full=5.2, local=940.9, msra=1499.3, gap_closed=-59.7%  [OK]
- musique: full=22.1, local=7611.9, msra=5596.2, gap_closed=26.6%  [OK]
- samsum: full=5.2, local=680.8, msra=612.2, gap_closed=10.2%  [OK]
- trec: full=19.5, local=4909.9, msra=1132.8, gap_closed=77.2%  [OK]
- lsht: full=17.7, local=7783.0, msra=7254.8, gap_closed=6.8%  [OK]
- lcc: full=36.8, local=9755.1, msra=20059.4, gap_closed=-106.0%  [OK]
- repobench-p: full=8.9, local=4896.0, msra=1134.7, gap_closed=77.0%  [OK]
- **Average gap closed: 6.9%**  → paper +6.9% **VERIFIED**
- **PROTOCOL ISSUE**: qcmsa arm evaluated on ~10x fewer tokens than local/full arms on ALL 15 tasks: gov_report (Q=3573 vs F=35415), qasper (Q=183 vs F=951), passage_retrieval_en (Q=27 vs F=150), narrativeqa (Q=135 vs F=312), qmsum (Q=764 vs F=3920), multi_news (Q=2549 vs F=13714), triviaqa (Q=21 vs F=206), hotpotqa (Q=29 vs F=221), 2wikimqa (Q=66 vs F=210), musique (Q=63 vs F=202), samsum (Q=189 vs F=1219), trec (Q=44 vs F=108), lsht (Q=13 vs F=226), lcc (Q=155 vs F=591), repobench-p (Q=44 vs F=663).
  Gap-closed numbers compare different eval subsets — must be disclosed in the paper and re-run with matched
  eval sets + official LongBench metrics in Phase 1.2.
  Metric caveat to add: gap closed computed in perplexity space, not official LongBench metrics.

## Table 5 (ablation)

- M=4, gate-on, L=4: params=22548, local@512=383.4, full=19.7, MSRA=361.7, gap_closed=6.0%  → paper params **WRONG (was 2×)**, gap values VERIFIED
- M=1, gate-off, L=4: params=9224, local@512=383.4, full=19.7, MSRA=361.6, gap_closed=6.0%  → paper params **WRONG (was 2×)**, gap values VERIFIED
- M=4, gate-off, L=4: params=21524, local@512=383.4, full=19.7, MSRA=361.7, gap_closed=6.0%  → paper params **WRONG (was 2×)**, gap values VERIFIED
- M=1, gate-off, L=8: params=18448, local@512=383.4, full=19.7, MSRA=351.6, gap_closed=8.7%  → paper params **WRONG (was 2×)**, gap values VERIFIED

## Table 6 (per-position, rebuilt from llama_per_pos_1024.json)

- n positions=1023; quartile stats (positive = MSRA NLL lower than local):
  - Q1: mean=+0.1812, %positive=69%
  - Q2: mean=+0.0241, %positive=77%
  - Q3: mean=+0.0287, %positive=91%
  - Q4: mean=+0.0292, %positive=91%
- positions 0-127 (inside window; far branch = 0 by construction → Δ should be exactly 0): mean=+0.1440, %positive=66%  **ARTIFACT — nonzero where theory says 0**
- **PROTOCOL ISSUE**: `src/eval/llama_per_position.py:185` evaluates on RANDOM token IDs (`torch.randint(100,50000)`), not natural text. Paper must disclose this or re-run on PG-19 text (recommended, cheap).
- Paper's old values (+0.008/+0.023/+0.031/+0.031, 100% positive) match NO results file → **UNTRACEABLE — REPLACE**.

## Table 7 (patch-layer budget)

- L=2: params=4612, local=561.4, full=15.3, MSRA=526.6, gap_closed=6.4%
- L=4: params=9224, local=561.4, full=15.3, MSRA=519.2, gap_closed=7.7%
- L=8: params=18448, local=561.4, full=15.3, MSRA=498.5, gap_closed=11.5%
(Params column 4,612/9,224/18,448 verified from n_spectral + code — correct.)

## Table 8 (systems)

- seq=512: local=39375, MSRA=4527, slowdown=8.70×, throughput_loss=89%, latency_increase=770% [paper local=34254]
- seq=1024: local=29471, MSRA=4370, slowdown=6.74×, throughput_loss=85%, latency_increase=574% [paper local=27083]
- seq=2048: local=20166, MSRA=4045, slowdown=4.98×, throughput_loss=80%, latency_increase=398% [paper local=17931]
- seq=4096: local=11535, MSRA=3188, slowdown=3.62×, throughput_loss=72%, latency_increase=262%  **JSON qcmsa run FAILED (=0); paper row (11535→3188) is from an older run, UNTRACEABLE in current JSON — re-measure in Phase 1.3**

## Cross-table consistency assertions (run as gate before any text edit)

- [PASS] T1@512 per-seed gap range == 11.2-13.3: actual 11.2-13.3
- [PASS] T1@1024 per-seed gap range == 6.9-7.5: actual 6.9-7.5
- [PASS] T6 M=1 gate-off L=8 == 8.7 (single run): actual 8.7
- [PASS] T7 L=8 == 11.5 (single run): actual 11.5
- [PASS] T7 L=8 (11.5) outside T1@1024 per-seed range (6.9-7.5): split-draw variance note justified
- [PASS] T4 average gap closed == 6.9: actual 6.9
- [PASS] param formula M4 gate-on L4 == 22548: formula gives 22548
- [PASS] param formula M1 gate-off L4 == 9224: formula gives 9224
- [PASS] param formula M4 gate-off L4 == 21524: formula gives 21524
- [PASS] param formula M1 gate-off L8 == 18448: formula gives 18448
- [PASS] T9(seq=512) v2 slowdown == 8.2: actual 8.19x
- [PASS] T9(seq=1024) v2 slowdown == 7.0: actual 6.98x
- [PASS] T9(seq=2048) v2 slowdown == 5.0: actual 4.98x
- [PASS] T9(seq=4096) recorded as OOM (honest replacement for untraceable row): status=OOM

**ALL PASS**

## Phase 1 assertions (fixed-implementation numbers)

- [PASS] R_M1L8 x3 canonical gap in 12.3-12.6: actual [12.54, 12.36, 12.54]
- [PASS] no cross-seed bit-identity under fp32 (sensitivity proven by s3): fp32 values [327.28741455078125, 327.7911071777344, 327.2366943359375]
  - note: fp16 fixed_split_eval.json retains s2=s4 bit-identity (expected quantization floor, appendix demo)
- [PASS] M1L4_s2 canonical gap in 5.2-5.5: actual 5.39
- [PASS] M1L4_s3 canonical gap in 5.2-5.5: actual 5.32
- [PASS] M4L4_s2 canonical gap in 8.4-8.8: actual 8.53
- [PASS] M4L4_s3 canonical gap in 8.4-8.9: actual 8.69
- [PASS] gamma-only@M1L8 canonical == 10.0 (was 9.5 in-loop caliber): actual 10.04
- [PASS] gamma-only@M4L8 canonical == 10.1: actual 10.09
- [PASS] gamma-only@M1L16 canonical == 26.4 (band A: 81-83% gamma share at both depths): actual 26.36, structure adds 5.4pp
- [PASS] M1L16 n=2 canonical 31.7 (MD-paper Table 2, seeds=2): actual [31.72, 31.73]
- [PASS] M4L8-gateoff n=2 canonical 15.4: actual [15.36, 15.36]
- [PASS] M4L4-gateoff n=2 canonical 6.1: actual [6.12, 6.14]
- [PASS] M4L8-gateoff fp16-identical pair discriminates under fp32: fp32 values [317.178955078125, 316.99615478515625]
- [PASS] G2 M1L8 x5 gain ~10.4 all positive: mean 10.4, [10.729798155408826, 10.90437353768861, 9.98429523940947, 9.80796027626079, 10.598729309099427]
- [PASS] LongBench official MSRA closure ~0 (|<2|): [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]
- [PASS] decode parity: six paired measurements median ~1.0 (FLOPs-identical prediction): ratios [1.1, 1.02, 1.01, 1.0, 1.0, 0.99], median 1.004
- [PASS] decode parity: range 0.9-1.15 across sessions: min 0.99 max 1.10
- [PASS] baseline_lora n=2 canonical 87.2-87.6 + wiring load_check clean: actual [87.39, 87.47], load_check_clean=True
- [PASS] baseline_dwconv n=2 canonical 78.0-78.5 + wiring load_check clean: actual [78.23, 78.32], load_check_clean=True
- [PASS] NIAH v5 R_M1L8_s2: near gc -15.2±0.5, mid gc>5, far |gc|<1.5, survival 1.0: near -15.23, mid +8.60, far +0.43, surv 1.0
- [PASS] NIAH v5 R_M4L4_s2: near gc -12.8±0.5, mid gc>5, far |gc|<1.5, survival 1.0: near -12.84, mid +7.50, far +0.03, surv 1.0
- [PASS] NIAH v5 R_M4L8_s2b: near gc -7.6±0.5, mid gc>5, far |gc|<1.5, survival 1.0: near -7.62, mid +8.40, far -0.42, surv 1.0
- [PASS] NIAH v5 R_M1L16_s2: near gc -4.3±0.5, mid gc>5, far |gc|<1.5, survival 1.0: near -4.32, mid +7.32, far -1.07, surv 1.0
- [PASS] NIAH v5 R_gamma_only_s2: near gc -14.3±0.5, mid gc>5, far |gc|<1.5, survival 1.0: near -14.26, mid +8.74, far +0.73, surv 1.0
- [PASS] NIAH v5 R_gammaonly_M1L16_s2: near gc -3.5±0.5, mid gc>5, far |gc|<1.5, survival 1.0: near -3.51, mid +8.17, far +0.85, surv 1.0
- [PASS] NIAH v5 M1L16 @1024 far gc ~+15 (vanishes @2048): actual 15.02
- [PASS] gate entropy ratio ~62.8% of uniform (selectivity): actual 0.628
- [PASS] gate query-perturbation response >> context perturbation (>10x): actual 112.9x
- [PASS] systems systems_llama_m1l8: @512 slowdown ~2.9x: actual 2.89x
- [PASS] systems systems_llama_m1l16: @512 slowdown ~4.8x: actual 4.84x
- [PASS] systems systems_llama_v2_m4l8: @512 slowdown ~8.7x: actual 8.71x
- [PASS] LongBench _kv: full mean 0.0804 reproduced exactly: actual 0.0804
- [PASS] LongBench _kv: local mean 0.033 (old collapse was artifact): actual 0.0332
- [PASS] LongBench _kv: qcmsa mean 0.033: actual 0.0334
- [PASS] LongBench _kv: closure <= 2% on every non-degenerate task: [('triviaqa', 0.0), ('gov_report', 1.28), ('qmsum', -0.0)]
- [PASS] LongBench _kv: window cost real on gov_report: full 0.137 vs local 0.003
- [PASS] RULER v2: full positive control (PPL<2.0, n>=30) + closure ~3.0% (<5): full_ppl 1.563, n 30, closure 2.98%
- [PASS] official PG-19 test partition: gain ~+10.1 (sec 5.1): actual +10.07%, gc 11.43% (file lacks seed provenance — robustness check only)

**ALL PHASE-1 ASSERTIONS PASS**
