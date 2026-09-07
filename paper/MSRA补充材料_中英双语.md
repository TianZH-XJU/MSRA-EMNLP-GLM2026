# MSRA 补充材料：实现审计、反转总账与可复现性档案

**本材料收录主论文全部数字的证据链：数据溯源索引、九项实现审计（A.1–A.9）、反转总表、种子与评估详情、以及机器执行的断言清单。主论文正文只呈现最终数据与结论；本档案记录它们是如何被验证到位的。**

---

## S1. 冻结声明与数据溯源

实验矩阵冻结于 tag `phase1-experiment-freeze`（commit `1bff7cdc` + 文字收尾 `4078888b`）。冻结点处，`paper/recompute_tables.py` 的 52 条断言全部通过（输出见 S5），单元测试 38 通过 / 1 失败（一条已文档化的 fp16 随机 token 溢出旧失败，与本工作无关）。冻结之后不再进行任何新实验，除非先有预登记协议与该实验所门禁的具体声明。

每个论文数字都可以从结果文件追溯到断言锚：

| 论文元素 | 数据文件（results/） | 断言锚 |
|---|---|---|
| 表 1：PG-19 困惑度（GPT-2 行） | G2_M1L8_s1..s5.json | P1-A8 |
| 表 1：LLaMA @512 行 | fixed_split_eval.json（16-chunk canonical） | P1-A1（×3 seeds） |
| 表 1：LLaMA @1024 行 | fixed_split_1024.json | S4（n=1 披露） |
| 表 2：阶梯（M1L8×3 / M1L4×2 / M4L4×2） | fixed_split_eval.json | P1-A1 / A2（双 seed） |
| 表 2：M1L16 / M4L8-gateoff / M4L4-gateoff | fixed_split_m1l16{,_s3} / gateoff_m4l8{,_s3} / gateoff_m4l4_{s2,s3}.json | P1-A2b（含 fp32 判别） |
| 表 2：仅增益三行 | fixed_split_gammaonly_{m1l8,m4l8,m1l16}.json | P1-A7（含 26.4% 断言） |
| 表 2：LoRA / dwconv | fixed_split_bl_{lora,dwconv}_{s2,s3}.json | P1-A11（含接线 load_check） |
| 表 2：未训练随机核 | fixed_split_eval.json（random 臂） | ④ 断言 |
| 表 3：NIAH v5（8 行） | niah_v5_ws128_all.json（6 ckpt，存活率 1.0） | P1-A12（六臂 + @1024 far） |
| 表 4：prefill 减速 | systems_llama_{m1l8,m1l16,v2_m4l8}.json（干净 session） | P1-A13 |
| 表 4：decode 列 | systems_llama_*_{kv,kv2}.json（两 session 配对） | P1-A10（median 1.004） |
| LongBench 段（0.0804/0.0332/0.0334） | longbench_official_{full,local,qcmsa}_kv.json | P1-A14 |
| RULER 3.0%（PPL 空间） | ruler_v2_2048_{full,local,qcmsa}_niah_single.json | P1-A5（阳性对照+闭合） |
| 门控选择性（62.8% / 113×） | gate_analysis_v2.json（文档化协议） | P1-A14b |
| §5.1 官方 test 复核（+10.1%） | pg19_official_test_3arm.json | P1-A15 |
| fp16 量化地板（S4） | v8_discrimination.json / v8_gateoff_fp32.json | P1-A1/A2b |
| 训练内 bit-identity 结案（A.5） | v8_roundtrip_final.json | （审计叙述） |

所有断言由 `paper/recompute_tables.py` 机器执行，完整输出存档于 `paper/REVISION_DATA_AUDIT.md`。

## S2. 实现审计（A.1–A.9）

本文在实验过程中发现并修复了九项实现缺陷。每项都遵循同一闭环：发现 → 取证 → 修复 → 撤回受影响结论 → 重测 → 断言钉住。

**A.1 核 FFT 缓存碰撞。** 推理路径的核 FFT 缓存 key 从 kernel 前 8 个元素导出；far mask 将这些元素全部置零，使得不同 σ 的核共享同一缓存条目。后果：训练期间可学习尺度 σ 实际未生效（全部 M=1 checkpoint 的 σ 冻结在初始化值 20.1），推理时不同配置的输出不可区分。修复后 σ 恢复可学习（训练协议下的真实轨迹很小：M1L8 各 seed 20.1→20.6–20.9，log σ 漂移 +0.02 至 +0.04；早期报告的"30 步漂移 0.33"出自单元测试的激进学习率，不属于训练协议）。

**A.2 空远位置噪声放大。** F(i)=∅（位置 i≤w）时 FFT 留 O(10⁻⁶) 噪声在分母 z 中；归一化 u = noise_C/(noise_z+ε) 将其放大至超过合法远信号（实测 in-window |y_far| 为 beyond-window 的 ~5 倍）。修复：F(i)=∅ 处硬零化 y_far（含分母路径）。

**A.3 NaN 梯度感染。** 宽 σ expert（σ 达 460）在 8 层配置下训练约 2,700 步后出现有限损失但非有限梯度；`clip_grad_norm_` 将 NaN 传播到全部参数。修复：total_norm isfinite 检查 + NaN-skip。

**A.4 NIAH 评估目标缺陷。** 早期 NIAH 协议的损失计算在 query 短语自身的续接 token 上（answer token 从未拼入输入），实际测量的是"needle 存在性对 query 短语 LM 的扰动"而非"needle 内容检索"。修复：answer token 拼入输入、损失窗口移至 answer 区域。（该修复下首次测得的近场改善本身又是截断 needle 的 artifact——被 A.9 取代。）

**A.5 训练内评估路径。** 训练内评估在三个 M1L8 seed 上产生 bit-identical 值（350.237183 × 3）。往返测试（独立进程、重建训练内 chunk 集、加载已存 checkpoint）在 fp16（350.240/350.304/350.275）与 fp32（350.268/350.297/350.289）下均给出三值互异的结果，且无一复现该记录值。这排除了两个候选解释：checkpoint 并不相同；fp16 输出舍入也不足以在重建 chunk 上将三者坍缩（量化地板现象真实存在——它使 canonical fp16 评测中 s2≡s4——但不延伸到训练内情形）。唯一自洽的结论是：训练内路径评估的状态与跨 seed 变化的训练参数无关；其确切根因在路径弃用前未被定位。论文全部数字不使用该路径：每个值都来自独立进程评测（`eval_fixed_split.py`）。

**A.6 基线适配器接线空转。** canonical（16-chunk）匹配基线评测的首次执行返回了与 local 臂 bit-identical 的 PPL（闭合 0.00%）：LoRA/dwconv wrapper 被正确构建且权重加载成功，但从未被赋值回注意力前向路径（训练 harness 做了这一步，评测 harness 最初漏做）。由于两类适配器都是零初始化，未接线的 wrapper 的输出与纯 local 注意力完全一致——这是 key 级 load 检查无法发现的静默空转，最终由"与 local 基线 16 位小数完全一致"这一反常暴露。修复：将 wrapper 接入前向路径，并在评测前断言适配器对象在前向路径上（阳性对照）。论文报告的是重评值：LoRA 87.4%（s2 87.39 / s3 87.47）、dwconv 78.3%（78.23/78.32）；此前流通的 LoRA 87.6% 来自已弃用的训练内路径。

**A.7 systems 配置错标。** 此前以"M1L8"名义呈现的预填充吞吐行，实际出自 systems harness 的 loader 默认结构——M=4、gate-on、未训练谱模块——而非所述配置。错标暴露于显式配置的 M1L16 测量（@512 4.8×）反而比"更小的 M1L8"（@512 8.2×）更快——结构上不可能。测量本身对它实际执行的结构有效（吞吐取决于 M 与 L，不依赖训练值），但所有派生表述（"5–8× 减速"）把默认配置的成本高估了约一倍。以显式结构 flag 重测、全部为同 session 配对：M1L8 2.0–2.9×、M1L16 3.0–4.8×、M4L8 5.0–8.7×——减速随远分支工作量（patch 层数 × expert 数：8/16/32 单元）单调放大，三种结构的解码均与 local 持平。

**A.8 解码路径延期修复。** 修复前的注意力 forward 接收 `past_key_values` 但从不写入 k/v（cache 写入代码位于被替换的 HF 注意力模块内部）。由于 cache 永不增长，HF `generate()`——信任 cache 对象的推进状态——每步把输入切成末 token：每个生成 token 只关注它自己，LongBench 生成坍缩为 gibberish（local/qcmsa 均值 ≈0.0001）。独立 decode 基准的手动循环向同一永不写入的 cache 每步只喂一个 token，因此此前报告的"decode = local"行测量的是 1-token 自注意力。两者是同一失效模式，既非真实路径也非可比路径。（我们的初始取证把 generate() 路径误判为"全前缀重喂"；前后对照数据否证了它——cached 与 full-recompute 解码已由单元测试证明等价，若旧路径是重喂，修复前后 local 分数应当相等，实测并不相等。）由 v3 代码审计发现。修复：在规范 forward 中实现真滑动窗口 KV cache——prefill 写入 k/v 且输出逐位不变（单元测试钉死）、decode 只关注缓存末尾 w+1 个 key、远分支结构性仅 prefill 生效（单元测试钉死）。修复后解码吞吐与 local 统计不可区分——按构造 FLOPs 相同；两个 session 的六次配对测量落于 0.99–1.10、中位数 1.004（表 4）。坏路径下计算的 LongBench 数字全部撤回；重跑精确复现 full 臂（均值 0.0804、逐 task 一致），并确认 local/修正臂此前的"生成坍缩"即上述 artifact；任务指标段落已改为修正后表述。重跑还暴露了每样本的分配器泄漏——谱模块按 padded 长度键控的 FP32 缓冲缓存在变长 prompt（triviaqa）上累积——已在模块层修复（缓存加上限）。

**A.9 needle 存活修复与近场反转。** NIAH 生成器先把 needle 插入全长 haystack、之后再为 query 与 answer 腾位截断；对近桶距离，第二次截断会静默切掉 needle 尾部——包括检索目标本身——使近桶混入不可答样本。在该协议下报告的近场改善（+20.6–40.2% gap-closed）正是在这些样本上测得，现全部撤回。修复：截断先于插入、needle 被钳位以 verbatim 落入最终 haystack、逐例断言存活且聚合率断言为 1.0（全部重跑实测 1.0）。对旧采样的确定性重放（逐位置对照已记录的 per-example 数据验证）量化了损害：11 枚近桶样本中 3 枚（27%）needle 尾部被截断；mid 与 far 均 0/11（几何预测：意图距离 ≲14 token 即截断，约占近桶 10–17%；3/11 为离散抽样下的实现值）。两点界定：截断几何受距离约束——mid 与 far 桶的 needle 在两个序列长度下都远离截断边界，其旧数字本就协议不变（这正是 @1,024 段落只需换锚无需重写的原因：+14.9% → +15.0% 的差即已披露的约 6 token 滑差）——需要撤回的只有 near 桶。且下述算术重构是示意性的而非独立检验（两个自由参数拟合一个方程）；承重证据是逐位置重放吻合与 3/11 对预测的一致。虚假改善的机制：以"…The secret key is XYZZY-"结尾的半截 needle 紧邻 query 是强分布外线索——设计上尖锐的 local 注意对它最脆弱（自信地错，NLL ≈8），而平滑化的远分支输出相对鲁棒——于是修正在恰好坏掉的样本上显得有帮助（3 枚坏样本 ≈8 加 8 枚完好样本 ≈1.6 恰好复现旧近桶均值 3.37）。元教训：没有运行时断言的"构造上保证"类论证一律不可信。修复后的协议下近场结论反转：全部配置使精确的窗口内检索轻微恶化（gap-closed −3.5% 至 −15.2%）、刚出窗口的预测有中等改善（+7–9%）、远场恢复仍 ≈0——修正在 local 失明处有帮助、在 local 尖锐处有害，这正是平滑器的签名；γ-only@L16 探针呈同一签名，把该效应系于增益通道。

## S3. 九环反转总表

论文中每一个幸存的数字都有不在场证明。九次，一个已宣称的结果被 harness 或协议缺陷推翻；每次推翻都遵循同一闭环——发现、取证、修复、撤回、重测、断言钉住——并留下可机器检查的锚。

| # | 被撤回的声明 | 根因 | 如何暴露 | 修复 | 断言锚 |
|---|---|---|---|---|---|
| 1 | 近场 "+26.5% @w=128"（初版投稿） | 一次崩溃的 NIAH run 的数字被误标为结果 | 溯源审计：表值不存在于 run 日志 | 诚实撤回；ws=256 重跑 | FORBIDDEN 清单 |
| 2 | "近场恶化"（v2 协议） | 损失算在 query 短语自身续接上——answer 从未入输入 | 代码审读损失切片 | answer 拼入；损失移至 answer 区（A.4） | v4/v5 协议；分区损失测试 |
| 3 | 近场 "+20.6–40.2%"（v4） | 二次截断静默切掉 needle 尾部——近桶混入不可答样本 | 代码包审查 + 几何复核 | 截断先于插入；逐例+聚合存活断言（A.9） | A12：六臂 v5 锚，存活率 1.0 |
| 4 | "多尺度无独立贡献" | 缺少 gate-off 对照时的过度泛化 | 2×2×2 网格的 gate-off 格 | 改写为"真实但次要（+0.7–3.0pp）" | 表 2 gate-off 格；A2b |
| 5 | "M4L8 不稳定" | NaN 梯度感染被误读为方法不稳定 | 守卫隔离实验 | total-norm isfinite 守卫 + NaN-skip；训练至 25.7% | A3；阶梯行 |
| 6 | "decode = local" | patched forward 从不写 KV cache → 基准测的是 1-token 自注意力 | 代码包审查（P0-1） | 真滑窗 KV cache；配对重测（A.8） | A10：六次配对，median 1.004 |
| 7 | LongBench "生成坍缩"（local ≈0.0001） | 同一 decode 缺陷——generate() 把输入切成末 token | 前后对照不对称（0.0001→0.0332，cached≡recompute 已测试钉死） | 修复路径下重跑；full 臂精确复现 | A14：closure ≤1.3%，full 0.0804 |
| 8 | "LoRA 87.6% / dwconv 78.3%" | 适配器 wrapper 从未接入前向路径（零初始化 ⇒ 静默空转 ≡ local） | canonical 首跑：与 local 基线 16 位小数一致 | 接线 + 阳性对照断言（A.6） | A11：87.4% / 78.3%，load_check 干净 |
| 9 | "Prefill 5–8×（M1L8）" | systems loader 默认值测的是 M4-gate-on 结构 | 显式 M1L16 反而比"更小的 M1L8"快——结构上不可能 | 显式结构 flag；同 session 配对（A.7） | A13：2.9×/4.8×/8.7× 随单元数缩放 |

同闭环、较小爆炸半径的卫生修复：γ-only@M1L8 9.5→10.0（训练内 vs canonical 口径）；表 3 嵌合体（旧 NLL + v4 百分比）按单一文件重建；6 任务与 7 任务均值统一；decode OOM 的堆碎片通过测量顺序重排解决；triviaqa 变长 prompt 的 padded-len 缓冲累积在模块层加上限；LongBench harness 的每样本谱缓存卫生。

## S4. 种子与评估详情

全部归因阶梯数字使用固定 16-chunk 评估集（SHA-256 前缀 8555f6e9）上的独立进程评测。fp16 评测下 M1L8 的 s2 与 s4 出现 bit-identical PPL（327.131…）；fp32 重评后三值互异（327.287/327.791/327.237），确认 bit-identity 为 fp16 量化地板，非 harness 缺陷。canonical 值在 fp32 重评下漂移 <0.05pp。同一现象也出现在 M4L8-gate-off：两 seed 在 fp16 下 bit-identical（317.125），fp32 下互异（317.179/316.996）。M1L16 两 seed 在 fp16 下即互异（259.107/259.043），其余各对亦互异。

各配置种子详情：M1L8 n=3（seed 2/3/4），M1L4 n=2（seed 2/3，canonical 5.39%/5.32%），M4L4 n=2（seed 2/3），M4L8 n=2（seed 2/3），M1L16 n=2（seed 2/3），M4L8-gate-off n=2（seed 2/3），M4L4-gate-off n=2（seed 2/3，canonical 6.12%/6.14%），LoRA n=2（seed 2/3），dwconv n=2（seed 2/3），GPT2-XL n=5（seed 1–5），LLaMA @1024 n=1（seed 2）。仅增益探针各 n=1（M1L8、M1L16、M4L8 均为 seed 2）。

## S5. 断言清单（机器执行输出，摘自 paper/REVISION_DATA_AUDIT.md）

```
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
- [PASS] R_M1L8 x3 canonical gap in 12.3-12.6: actual [12.54, 12.36, 12.54]
- [PASS] no cross-seed bit-identity under fp32 (sensitivity proven by s3): fp32 values [327.287, 327.791, 327.237]
- [PASS] M1L4_s2 canonical gap in 5.2-5.5: actual 5.39
- [PASS] M1L4_s3 canonical gap in 5.2-5.5: actual 5.32
- [PASS] M4L4_s2 canonical gap in 8.4-8.8: actual 8.53
- [PASS] M4L4_s3 canonical gap in 8.4-8.9: actual 8.69
- [PASS] gamma-only@M1L8 canonical == 10.0 (was 9.5 in-loop caliber): actual 10.04
- [PASS] gamma-only@M4L8 canonical == 10.1: actual 10.09
- [PASS] gamma-only@M1L16 canonical == 26.4 (band A: 81-83% gamma share at both depths): actual 26.36, structure adds 5.4pp
- [PASS] M1L16 n=2 canonical 31.7 (Table 2, seeds=2): actual [31.72, 31.73]
- [PASS] M4L8-gateoff n=2 canonical 15.4: actual [15.36, 15.36]
- [PASS] M4L4-gateoff n=2 canonical 6.1: actual [6.12, 6.14]
- [PASS] M4L8-gateoff fp16-identical pair discriminates under fp32: fp32 values [317.179, 316.996]
- [PASS] G2 M1L8 x5 gain ~10.4 all positive: mean 10.4
- [PASS] LongBench official MSRA closure ~0 (|<2|): [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]
- [PASS] decode parity: six paired measurements median ~1.0 (FLOPs-identical prediction): ratios [1.1, 1.02, 1.01, 1.0, 1.0, 0.99], median 1.004
- [PASS] decode parity: range 0.9-1.15 across sessions: min 0.99 max 1.10
- [PASS] baseline_lora n=2 canonical 87.2-87.6 + wiring load_check clean: actual [87.39, 87.47]
- [PASS] baseline_dwconv n=2 canonical 78.0-78.5 + wiring load_check clean: actual [78.23, 78.32]
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
- [PASS] official PG-19 test partition: gain ~+10.1 (sec 5.1): actual +10.07%, gc 11.43%
**ALL ASSERTIONS PASS**
```

---
---


# MSRA Supplementary Material: Implementation Audit, Reversal Ledger, and Reproducibility Archive

**This document records the evidence chain behind every number in the main paper: the data provenance index, nine implementation-audit entries (A.1–A.9), the reversal ledger, seed and evaluation details, and the machine-executed assertion list. The main text presents only the final data and conclusions; this archive documents how they were verified into place.**

---

## S1. Freeze Declaration and Data Provenance

The experiment matrix is frozen at tag `phase1-experiment-freeze` (commit `1bff7cdc`, plus text-level close-out `4078888b`). At the freeze point, all 52 assertions in `paper/recompute_tables.py` pass (output in S5), and the unit-test suite stands at 38 passed / 1 failed (one documented legacy failure: fp16 overflow on random tokens, unrelated to this work). After the freeze, no new experiment is permitted without a pre-registered protocol and a named claim it gates.

Every number in the paper can be traced from a results file to an assertion anchor:

| Paper element | Data file (results/) | Assertion anchor |
|---|---|---|
| Table 1: PG-19 perplexity (GPT-2 row) | G2_M1L8_s1..s5.json | P1-A8 |
| Table 1: LLaMA @512 row | fixed_split_eval.json (16-chunk canonical) | P1-A1 (×3 seeds) |
| Table 1: LLaMA @1024 row | fixed_split_1024.json | S4 (n=1 disclosed) |
| Table 2: ladder (M1L8×3 / M1L4×2 / M4L4×2) | fixed_split_eval.json | P1-A1 / A2 (both seeds) |
| Table 2: M1L16 / M4L8-gateoff / M4L4-gateoff | fixed_split_m1l16{,_s3} / gateoff_m4l8{,_s3} / gateoff_m4l4_{s2,s3}.json | P1-A2b (incl. fp32 discrimination) |
| Table 2: three gain-only rows | fixed_split_gammaonly_{m1l8,m4l8,m1l16}.json | P1-A7 (incl. the 26.4% assertion) |
| Table 2: LoRA / dwconv | fixed_split_bl_{lora,dwconv}_{s2,s3}.json | P1-A11 (incl. wiring load_check) |
| Table 2: untrained random kernel | fixed_split_eval.json (random arm) | ④ assertion |
| Table 3: NIAH v5 (8 rows) | niah_v5_ws128_all.json (6 ckpts, survival rate 1.0) | P1-A12 (six arms + @1024 far) |
| Table 4: prefill slowdowns | systems_llama_{m1l8,m1l16,v2_m4l8}.json (clean session) | P1-A13 |
| Table 4: decode column | systems_llama_*_{kv,kv2}.json (two paired sessions) | P1-A10 (median 1.004) |
| LongBench paragraph (0.0804/0.0332/0.0334) | longbench_official_{full,local,qcmsa}_kv.json | P1-A14 |
| RULER 3.0% (PPL space) | ruler_v2_2048_{full,local,qcmsa}_niah_single.json | P1-A5 (positive control + closure) |
| Gate selectivity (62.8% / 113×) | gate_analysis_v2.json (documented protocol) | P1-A14b |
| §5.1 official-test replication (+10.1%) | pg19_official_test_3arm.json | P1-A15 |
| fp16 quantization floor (S4) | v8_discrimination.json / v8_gateoff_fp32.json | P1-A1/A2b |
| In-loop bit-identity case closure (A.5) | v8_roundtrip_final.json | (audit narrative) |

All assertions are machine-executed by `paper/recompute_tables.py`; the full output is archived in `paper/REVISION_DATA_AUDIT.md`.

## S2. Implementation Audit (A.1–A.9)

Nine implementation defects were discovered and repaired over the course of this work. Each followed the same closed loop: discovery → forensics → repair → withdrawal of affected claims → re-measurement → assertion pin.

**A.1 Kernel-FFT Cache Collision.** The inference-path kernel FFT cache key was derived from the first 8 kernel elements; the far mask zeroes these for every σ, causing all configurations to share a single cached entry. Consequence: the learnable scale σ was effectively frozen at initialization (all M=1 checkpoints have σ ≡ 20.1); the original "learnable multiscale" never engaged. After the fix, σ trains again (the true trajectory under the training protocol is small: M1L8 seeds move 20.1→20.6–20.9, a log-σ drift of +0.02 to +0.04; an earlier report of "0.33 drift in 30 steps" came from a unit test's aggressive learning rate, not from the training protocol).

**A.2 Noise Amplification at Empty-Far Positions.** At positions i ≤ w (empty far context), FFT leaves O(10⁻⁶) noise in the denominator z; the normalization u = noise_C/(noise_z+ε) amplifies this above the legitimate far signal (measured in-window |y_far| was ~5× the beyond-window value). Fix: hard-zero y_far where F(i) = ∅, including the denominator pathway.

**A.3 NaN Gradient Infection.** Wide-σ experts (σ up to 460) produced finite losses with non-finite gradients roughly 2,700 steps into training in the 8-layer configuration; `clip_grad_norm_` propagated the NaN into every parameter. Fix: a total-norm isfinite check plus NaN-skip.

**A.4 NIAH Evaluation Target Defect.** The early NIAH protocol computed loss on the query phrase's own continuation (answer tokens were never appended to the input), measuring "the effect of needle presence on the query phrase's language modeling" rather than needle content retrieval. Fix: answer tokens appended to the input, loss window moved to the answer region. (The near-field improvement first measured under this fix was itself an artifact of truncated needles — superseded by A.9.)

**A.5 Training-Internal Evaluation Path.** Training-internal evaluation produced bit-identical values across the three M1L8 seeds (350.237183 × 3). A round-trip test — fresh process, reconstruction of the in-loop chunk set, loading each saved checkpoint — yields three distinct values under both fp16 (350.240/350.304/350.275) and fp32 (350.268/350.297/350.289), and reproduces none of the recorded value. This rules out the two candidate explanations: the checkpoints are not identical, and fp16 output rounding does not collapse them on these chunks (the quantization floor is real — it collapses s2/s4 in the canonical fp16 evaluation — but does not extend to the in-loop case). The only self-consistent conclusion is that the in-loop path evaluated a state independent of the seed-varying trained parameters; its exact root cause was not identified before the path was deprecated. No paper number uses this path: every value comes from independent-process evaluation (`eval_fixed_split.py`).

**A.6 Baseline Adapter Wiring No-Op.** The first execution of the canonical (16-chunk) matched-baseline evaluation returned PPL bit-identical to the local arm (0.00% closure): the LoRA/dwconv wrappers were constructed and their weights loaded, but never assigned into the attention forward path (the training harness performs this assignment; the evaluation harness initially did not). Because both adapter types are zero-initialized, an unwired wrapper reproduces local attention exactly — a silent no-op that no key-level load check can catch. The 16-decimal-digit match with the local baseline is what exposed it. Fix: wire the wrapper into the forward path and assert the adapter object is present in the forward path (positive control) before evaluating. The reported values are the re-evaluated ones: LoRA 87.4% (s2 87.39 / s3 87.47), dwconv 78.3% (78.23/78.32); the previously circulated LoRA 87.6% derived from the deprecated in-loop path.

**A.7 Systems Configuration Mislabeled.** The prefill-throughput rows previously presented as the default M1L8 were produced by the systems harness's loader default — M=4, gate-on, untrained spectral module — not the stated configuration. The mislabel surfaced when an explicitly-configured M1L16 measurement (4.8× @512) proved faster than the supposedly smaller M1L8 (8.2× @512), which is structurally impossible. The measurement itself was valid for the structure it actually exercised (throughput depends on M and L, not on trained values), but every derived claim ("5–8× slowdown") overstated the default configuration's cost by roughly a factor of two. Re-measured with explicit structure flags, all three configurations in matched same-session pairs: M1L8 2.0–2.9×, M1L16 3.0–4.8×, M4L8 5.0–8.7× — the slowdown scales with far-branch work (patched layers × experts: 8/16/32 units), and decode is at local parity for all three.

**A.8 Decode Path Deferred-Fix.** The pre-fix attention forwards accepted `past_key_values` but never wrote k/v (the cache-write code lives in the HF attention module we replace). Because the cache never grew, HF `generate()` — trusting the cache object's progression — sliced its input to the last token at every step: each generated token attended only itself, and the LongBench generations collapsed to gibberish (local/qcmsa means ≈0.0001). The standalone decode benchmark's manual loop fed a single token against the same never-written cache, so the previously reported "decode = local" rows measured 1-token self-attention. Neither was a real or comparable path; both are the same failure mode. (Our initial forensic reading attributed the generate() path to full-prefix re-feeding; the before/after data refutes this — cached and full-recompute decode are equivalent by unit test, so a re-feeding path would have reproduced the corrected scores, and it did not.) Surfaced by a code audit. Fix: a true sliding-window KV cache in the canonical forwards — prefill writes k/v with bitwise-unchanged outputs (unit-test pinned), decode attends the last w+1 cached keys, and the far branch is structurally prefill-only (unit-test pinned). Post-fix decode throughput is statistically indistinguishable from local — FLOPs identical by construction; six paired measurements across two sessions span 0.99–1.10 with median 1.004 (Table 4). The LongBench numbers computed under the broken path are withdrawn; the re-run reproduces the full arm exactly (mean 0.0804, per-task identical) and confirms the local/corrected arms' earlier "generation collapse" was the artifact described above; the task-metric paragraph states the corrected form. The re-run also exposed a per-sample allocator leak — the spectral module's padded-length-keyed FP32 buffer cache accumulating on variable-length prompts (triviaqa) — fixed at the module level (bounded cache).

**A.9 Needle-Survival Repair and the Near-Field Reversal.** The NIAH generator inserted the needle into the full-length haystack and only then truncated the haystack to make room for query and answer; for near-bucket distances the second truncation silently cut the needle tail — including the retrieval target — seeding the near bucket with unanswerable samples. The near-field improvements reported under that protocol (+20.6–40.2% gap closed) were measured on such samples and are withdrawn. Fix: truncation now precedes insertion, the needle is clamped to fit the final haystack verbatim, survival is asserted per example and the aggregate rate is asserted to be 1.0 (measured rate 1.0 across all re-runs). A deterministic replay of the old sampling — validated position-by-position against the recorded per-example data — quantifies the damage: 3 of 11 near-bucket samples (27%) had the needle tail truncated; 0 of 11 mid, 0 of 11 far (geometric prediction: cut when the intended distance ≲ 14 tokens, i.e. ~10–17% of the near bucket; the realized 3/11 reflects discrete sampling). Two scope notes. The truncation geometry is distance-bounded — mid- and far-bucket needles sit far from the truncation frontier at both sequence lengths, so their old numbers were protocol-invariant (which is why the @1,024 figures needed only re-anchoring, not rewriting: +14.9% → +15.0% differs by the disclosed ~6-token slide) — only the near bucket required withdrawal. And the arithmetic reconstruction below is illustrative, not an independent check (two free parameters, one equation); the load-bearing evidence is the position-by-position replay match and the 3/11-versus-prediction agreement. The mechanism of the spurious improvement: a half-needle ending "…The secret key is XYZZY-" adjacent to the query is a strong out-of-distribution cue — local attention, sharp by design, is the most vulnerable to it (confidently wrong, NLL ≈8), while the smoothed far-branch output is comparatively robust — so the correction appeared helpful precisely on the broken samples (3 broken samples at ≈8 plus 8 intact samples at ≈1.6 reproduce the old near-bucket mean of 3.37 exactly). Meta-lesson: no "guaranteed by construction" argument is trustworthy without a runtime assertion. Under the repaired protocol the near-field conclusion reverses: every configuration slightly degrades precise in-window retrieval (−3.5% to −15.2% gap closed), moderately improves just-beyond-window prediction (+7–9%), and leaves far-field recovery at ≈0 — the correction helps where local attention is blind and hurts where it is sharp, which is exactly the signature of a smoother; the gain-only@L16 probe carries the same signature, tying the effect to the gain channel.

## S3. The Reversal Ledger

Every surviving number in this paper has an alibi. Nine times, a claimed result was overturned by a harness or protocol defect; each overturning followed the same loop — discovery, forensics, repair, withdrawal, re-measurement, assertion pin — and each left a machine-checkable anchor behind.

| # | Withdrawn claim | Root cause | How it surfaced | Repair | Assertion anchor |
|---|---|---|---|---|---|
| 1 | Near-field "+26.5% @w=128" (original submission) | a crashed NIAH run's numbers were mislabeled as results | provenance audit: table values absent from run logs | honest withdrawal; ws=256 re-run | FORBIDDEN list |
| 2 | "Near-field degradation" (v2 protocol) | loss computed on the query phrase's own continuation — the answer never entered the input | code review reading the loss slice | answer appended; loss on the answer region (A.4) | v4/v5 protocol; per-region loss test |
| 3 | Near-field "+20.6–40.2%" (v4) | a second haystack truncation silently cut the needle tail — the near bucket was seeded with unanswerable samples | package review + geometric re-check | truncation before insertion; per-example + aggregate survival asserts (A.9) | A12: six v5 anchors, survival 1.0 |
| 4 | "Multiscale has no independent contribution" | over-generalization without the gate-off control | gate-off cells in the 2×2×2 grid | reworded "real but secondary (+0.7–3.0pp)" | Table 2 gate-off cells; A2b |
| 5 | "M4L8 is unstable" | NaN gradient infection misread as method instability | guard-isolation experiment | total-norm isfinite guard + NaN-skip; trains to 25.7% | A3; ladder row |
| 6 | "decode = local" | patched forward never wrote the KV cache → the benchmark measured 1-token self-attention | package review (P0-1) | true sliding-window KV cache; paired re-measurement (A.8) | A10: six paired measurements, median 1.004 |
| 7 | LongBench "generation collapse" (local ≈0.0001) | the same decode defect — generate() sliced input to the last token | before/after asymmetry (0.0001→0.0332, with cached≡recompute test-pinned) | re-run under the fixed path; the full arm reproduces exactly | A14: closure ≤1.3%, full 0.0804 |
| 8 | "LoRA 87.6% / dwconv 78.3%" | adapter wrappers never wired into the forward path (zero-init ⇒ silent no-op ≡ local) | canonical first run: 16-decimal match with the local baseline | wiring + positive-control assert (A.6) | A11: 87.4% / 78.3%, load_check clean |
| 9 | "Prefill 5–8× (M1L8)" | the systems loader default measured the M4-gate-on structure | explicit M1L16 measured faster than the "smaller" M1L8 — structurally impossible | explicit structure flags; same-session pairs (A.7) | A13: 2.9×/4.8×/8.7× scaling with units |

Same-loop hygiene fixes with smaller blast radius: γ-only@M1L8 9.5→10.0 (in-loop versus canonical caliber); a Table 3 chimera (old NLLs with v4 percentages) rebuilt from a single file; 6-task versus 7-task means unified; decode OOM from heap fragmentation solved by reordering the measurement; triviaqa's padded-length buffer accumulation bounded inside the module; per-sample spectral-cache hygiene in the LongBench harness.

## S4. Seed and Evaluation Details

All attribution-ladder values use the fixed 16-chunk evaluation set (SHA-256 prefix 8555f6e9) with independent-process evaluation. Under fp16 evaluation, M1L8 seeds s2 and s4 show bit-identical PPL (327.131…); fp32 re-evaluation yields three distinct values (327.287/327.791/327.237), confirming the bit-identity is an fp16 quantization floor, not a harness defect. Canonical values drift <0.05pp under fp32 re-evaluation. The same phenomenon appears for M4L8-gate-off: the two seeds are bit-identical under fp16 (317.125) but distinct under fp32 (317.179/316.996). M1L16 seeds are distinct already under fp16 (259.107/259.043), as are all remaining pairs.

Seed details: M1L8 n=3 (seeds 2/3/4), M1L4 n=2 (seeds 2/3; canonical 5.39%/5.32%), M4L4 n=2 (seeds 2/3), M4L8 n=2 (seeds 2/3), M1L16 n=2 (seeds 2/3), M4L8-gate-off n=2 (seeds 2/3), M4L4-gate-off n=2 (seeds 2/3; canonical 6.12%/6.14%), LoRA n=2 (seeds 2/3), dwconv n=2 (seeds 2/3), GPT2-XL n=5 (seeds 1–5), LLaMA @1024 n=1 (seed 2). Gain-only probes: n=1 each (M1L8, M1L16, M4L8 — all seed 2).

## S5. Assertion List (Machine-Executed Output, from paper/REVISION_DATA_AUDIT.md)

The complete assertion output is reproduced in Section S5 of the Chinese half above; the two halves share one identical machine log.
