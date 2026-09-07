# MSRA — Multi-scale Spectral Residual Attention: Code, Data, and Paper

**A parameter-efficient spectral correction for local attention — effective and failure boundaries, with every number assertion-anchored.**

*轻量谱修正（MSRA）的完整实验发布：代码、评估数据与中英双语论文。全部数字由机器断言锚定。*

License / 许可证：**CC BY 4.0** (see [`LICENSE`](LICENSE))

---

## What this repository contains / 本仓库内容

This is the complete, frozen release accompanying the paper *"MSRA: A Parameter-Efficient Spectral Correction for Local Attention — Where It Helps, and Where It Fails"* and its bilingual supplementary material. It contains the final implementation, the fixed evaluation protocol, all result data backing every number in the paper, and a machine-executed assertion suite that re-derives those numbers from the data files.

本仓库是论文《MSRA：局部注意力的参数高效谱修正——有效边界与失败边界》及其中英双语补充材料的完整冻结发布，包含最终实现、固定评估协议、支撑论文全部数字的结果数据，以及可将这些数字从数据文件中重新推导出来的机器断言套件。

**Headline results / 主要结果** (LLaMA 3.2-1B, frozen base, w=128):

- Spectral correction closes up to **31.7%** of the local–full perplexity gap; 16 gain-only scalars alone close **26.4%** / 谱修正最多闭合 31.7% 的困惑度差距；仅 16 个增益标量即闭合 26.4%
- Budget-matched LoRA closes **87.4%** — the gap is mostly generic capacity / 同预算 LoRA 闭合 87.4%——差距主体是通用容量
- The correction is a statistical smoother: it helps just beyond the window (+7–9% NLL), slightly costs in-window precision, and recovers ≈0 at far distances / 修正是统计平滑器：刚出窗口 +7–9%，窗口内轻微代价，远场 ≈0
- Prefill slows 2.0–2.9× (default) to 8.7× (widest multi-scale); decode is statistically indistinguishable from local / 预填充减速 2.0–2.9×（默认）至 8.7×（最宽多尺度）；解码与局部注意力统计不可区分

## Repository layout / 仓库结构

```
├── README.md                        this file / 本文件
├── LICENSE                          CC BY 4.0 full legal code
├── requirements.txt                 Python environment
├── paper/
│   ├── MSRA论文_终版_中英双语.md      main paper (ZH + EN)
│   ├── MSRA补充材料_中英双语.md       supplementary: audit A.1–A.9, reversal ledger, assertions
│   ├── REVERSAL_LEDGER.md           nine-reversal compact table
│   ├── REVISION_DATA_AUDIT.md       full machine assertion output (52 PASS / 0 FAIL at freeze)
│   └── recompute_tables.py          the assertion suite (re-derives every paper number)
├── src/
│   ├── kernels/correct_spectral_attention.py   the spectral kernel (FFT Toeplitz conv; guarded)
│   ├── models/llama_forwards.py                canonical patched forwards (true sliding-window KV cache)
│   └── eval/                                   all experiment harnesses (see "Reproducing")
├── tests/                           contract tests (KV-cache 7/7; forward parity)
└── results/                         every result file the assertions read (JSON)
```

## Environment / 环境

Python 3.12, PyTorch 2.9.1 (CUDA), Transformers 5.5.4. One 24 GB GPU (all experiments ran on RTX 4090 24GB). External model weights and datasets are **not** redistributed — see "External assets" below.

```bash
pip install -r requirements.txt
```

External assets you must provide yourself / 需自行准备的外部资产:

| Asset | Source | Used by |
|---|---|---|
| LLaMA 3.2-1B weights | Meta (Hugging Face: `meta-llama/Llama-3.2-1B`) | all LLaMA experiments |
| GPT2-XL weights | OpenAI (Hugging Face: `gpt2-xl`) | GPT-2 replication row |
| PG-19 corpus | DeepMind (Hugging Face: `deepmind/pg19`) | language modeling, NIAH haystack |
| LongBench | THUNLP (`THUDM/LongBench`) | task metrics (7 English tasks) |
| RULER | NVIDIA (`bosonai/RULER` subset used: `niah_single_2k`) | retrieval control |

Point the harnesses at your local copies: the default paths are `../basemodel/llama3.2-1b` and `../dataset/{pg19,longbench}` relative to the repository root (constants `MP`, `PG19`, `LB` at the top of the eval modules).

## Verify in 30 seconds / 30 秒验证

```bash
# 1. every number in the paper, re-derived from the released data files:
python paper/recompute_tables.py
#    → prints the full audit; must end with **ALL PHASE-1 ASSERTIONS PASS** (52 PASS / 0 FAIL)

# 2. KV-cache contract tests (CPU-only):
python -m pytest tests/test_llama_kv_cache.py -v
#    → 7 passed (prefill bitwise-unchanged; cached decode ≡ full-recompute; far branch prefill-only; ...)
```

If both pass, the released data and the released code reproduce every table in the paper — no GPU required for this step.

## Reproducing the experiments / 复现实验

All commands run from the repository root with a GPU selected, e.g. `CUDA_VISIBLE_DEVICES=0`. Thread caps (`OMP_NUM_THREADS=8`) are recommended when running several jobs concurrently.

**1. Fixed evaluation split** (the fixed 16-chunk exam every configuration shares):

```bash
python -m src.eval.make_eval_split          # writes results/eval_split_256.json (SHA-256 recorded)
```

**2. Training a correction configuration** (3,000 steps, ~1–4 h on one 4090):

```bash
# default spectral correction: M=1, gate-off, L=8, w=128
python -m src.eval.llama_train --exp_name R_M1L8_s2 --train_len 512 --steps 3000 \
    --window 128 --experts 1 --stationary --patch_layers 8 --seed 2 \
    --eval_manifest results/eval_split_256.json
# gain-only probe:   add --train_gamma_only
# depth sweep:       --patch_layers 4 / 16
# multi-scale:       --experts 4 (add --gate_off for the gate-off control)
```

**3. Canonical evaluation** (independent process, 16 chunks — this is where all ladder numbers come from):

```bash
python -m src.eval.eval_fixed_split --ckpts results/R_M1L8_s2.pt \
    --out results/fixed_split_eval.json
# budget-matched adapters are trained by src/eval/matched_baselines.py, then evaluated with
#   --ckpts "baseline:lora:results/R_bl_lora_s2.pt"
```

**4. Needle-in-a-haystack** (survival-asserted protocol, audit A.9):

```bash
python -m src.eval.niah_grid --model_family llama --seq_lens 1024,2048 --window_sizes 128 \
    --models full,local,qcmsa --ckpt results/R_M1L8_s2.pt --output_dir /tmp/niah
```

**5. Official LongBench metrics** (7 English tasks, 2,048-token prompts):

```bash
python -m src.eval.longbench_official --models full,local,qcmsa \
    --max_prompt_tokens 2048 --out results/longbench_official_kv.json
```

**6. RULER control / systems benchmark / gate selectivity**:

```bash
python -m src.eval.ruler_eval --task niah_single --models full,local,qcmsa
python -m src.eval.systems_benchmark --model llama --patch_layers 8 --num_experts 1 --gate_off --suffix _m1l8
python -m src.eval.gate_entropy_analysis --ckpt results/R_M4L8_s2b.pt --out results/gate_analysis_v2.json
```

The exact commands, seeds, and provenance (git hashes, checkpoint SHA-256, manifest SHA-256) for every released result file are embedded in the `provenance` blocks of the `results/*.json` training records and documented in the supplementary material (S1–S5).

## What is in `results/` / 结果数据说明

- `eval_split_256.json` — the frozen evaluation manifest (256 books outside the training pool; SHA-256 `8555f6e9…`)
- `fixed_split_*.json` — canonical 16-chunk evaluations for every ladder configuration (both seeds where n=2)
- `niah_v5_ws128_all.json` — the survival-repaired NIAH protocol (6 checkpoints; per-example records; survival rate 1.0)
- `systems_llama_*_{kv,kv2}.json` — prefill/decode throughput, two independent paired sessions
- `longbench_official_*_kv.json`, `ruler_v2_2048_*.json` — task-metric evidence
- `gate_analysis_v2.json` — gate selectivity (entropy ratio, perturbation response; documented protocol)
- `v8_*.json` — precision-discrimination forensics (fp16 quantization floor, round-trip test)
- `R_*.json`, `G2_*.json` — training records (config, per-seed histories, provenance)
- legacy-era files (`best_mix_*`, `llama_ws*`, `cfg_*`, …) are included only because the assertion suite cross-checks them; they belong to the pre-fix era and are never cited as current evidence

## Checkpoints / 模型权重

Trained correction checkpoints (`results/*.pt`, ~100 MB each) are **not** included in this release to keep it lightweight. They are available from the authors on reasonable request.

## License / 许可证

Everything in this repository — code, evaluation data, result files, and both paper texts — is released under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. You may share and adapt it for any purpose, provided you give appropriate credit and do not apply additional restrictions. See [`LICENSE`](LICENSE) for the full legal code.

External model weights and datasets listed above remain under their own licenses and terms.

## Citation / 引用

```bibtex
@misc{msra2026,
  title  = {MSRA: A Parameter-Efficient Spectral Correction for Local Attention --
            Where It Helps, and Where It Fails},
  author = {The MSRA Authors},
  year   = {2026},
  url    = {<this repository>},
  license = {CC-BY-4.0}
}
```

> Before publishing, replace `The MSRA Authors` in `LICENSE` and the citation block with the final author list.
> 发布前请将 `LICENSE` 与引用条目中的 "The MSRA Authors" 替换为正式作者名单。

## Notes on reproducibility / 可复现性说明

- All attribution-ladder numbers come from **independent-process** evaluations on the fixed 16-chunk split; training-internal evaluations are deprecated (audit A.5).
- fp16 evaluation shows a quantization floor (bit-identical outputs across genuinely different checkpoints); fp32 re-evaluation discriminates all seeds and moves canonical values by <0.05pp (supplementary S4).
- The nine implementation defects discovered and repaired during this work — and the nine claim reversals they caused — are documented in the supplementary material (S2, S3). Every surviving number carries an assertion anchor.
- 本发布包自带的测试套件（`tests/`，18 条，纯 CPU）全部通过；开发仓的全套测试在冻结点为 38 passed / 1 failed，唯一失败是一条已文档化的历史遗留问题（fp16 随机 token 溢出），与本文任何声明无关。
- The release's own test suite (`tests/`, 18 tests, CPU-only) passes in full; the full development suite stood at 38 passed / 1 failed at the freeze point (the one failure is a documented legacy fp16 random-token overflow, unrelated to any claim).
