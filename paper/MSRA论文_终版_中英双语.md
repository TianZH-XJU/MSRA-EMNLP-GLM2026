# MSRA：局部注意力的参数高效谱修正——有效边界与失败边界

**一项对轻量谱修正的系统归因研究**

---

## 摘要

让大语言模型读长文，最贵的环节是注意力：每个词若都要回头核对前文全部内容，计算量便随文章长度平方式增长。工程上最常用的减法是滑动窗口——每个词只被允许看见最近的一段，成本随之降到与长度成正比，代价是窗口之外的内容被彻底遗忘。本文追问一个直接的问题：如果模型已经训练完毕、原始权重不许改动、新增预算只有几万个参数，我们还能把丢失的远方信息补回来吗？补回来的又是什么？

我们给出多尺度谱残差注意力（MSRA）：在冻结模型的最后若干层上，为注意力输出并联一条"远眺支路"——用一个随距离平滑衰减的权重，把窗口之外的全部历史浓缩成一份补充摘要，加回原有表示。整条支路只含约九千到四万五千个可训练参数，另有一组 8 至 16 个标量的极简探针用于归因。在 PG-19 长篇小说语料上，这条支路最多把局部注意力与全局注意力之间的困惑度差距补回 31.7%（LLaMA 3.2-1B，十六层配置），增益在两种模型架构、多个随机种子下稳定为正，并在官方测试分区上复核成立。

真正有价值的发现藏在归因里。收益的两个主要来源都出人意料地朴素：一个是每层一个可学习的"输出强度"旋钮——仅训练这 16 个标量就能补回 26.4%；另一个是把支路铺到更深的层，收益随深度单调加速。精心设计的核结构与门控只有次要贡献，而同预算的通用低秩适配器 LoRA 能补回 87.4%，说明这个差距的主体是通用容量不足，而非缺少一条专门的远方通道。检索实验进一步揭示了修正的本质：它是一个统计平滑器，让刚出窗口的预测变好，让窗口内的精确答案预测略微变差，对真正的远距离检索无能为力，也不迁移到长文任务指标。效率上，默认配置使预填充减慢 2 至 3 倍，解码则完全不受影响。本文为"轻量谱修正能补什么、补不了什么"提供了一幅完整、可复现的边界图，也为长上下文模型的设计提供了直接的实证参照。

---

## 1. 引言

现代大语言模型处理长序列的主流方案之一，是让每个词只关注最近的 w 个词。这个设计把计算量从随长度平方增长压到线性增长，使得几千甚至上万词的文章可以被实际处理。但每一个读过小说的人都知道第三章埋下的伏笔有多重要——而对只看得见窗口的模型来说，三章之前的内容等于从未存在。

已有大量工作试图补上这个缺口：稀疏注意力让部分词看得更远，线性注意力用核近似替代标准注意力，状态空间模型换了一条完全不同的记忆路线，局部-全局混合架构则在窗口之外再加一条全局通路。这些方案大多要替换注意力机制本身，或者从头到尾重新训练模型。然而有些场景没有这样的奢侈——模型已经部署上线，原始权重不能触碰，允许改动的参数只有几万个。我们的问题由此而来：在这样苛刻的限制下，一条谱形式的远场修正支路，究竟能补回多少丢失的信息？补回的部分又来自哪里？

我们提出 MSRA：把一条基于频域卷积的残差支路嫁接到冻结模型的最后 L 层，只训练这条支路。这个设定既有明确的应用意义——无法重训的部署模型、预算极低的适配场景——也构成了一张干净的归因实验台，因为支路的每个零件（输出增益、核结构、多尺度、门控、嫁接深度）都可以单独打开或关闭，逐个接受检验。

我们带着这张实验台做了十三种配置的系统测量，得到的图景比预期清晰得多。困惑度的改善真实存在且跨种子稳定，但它的来源并非我们设计的谱结构本身，而是两个更朴素的因素：每层一个可学习的输出强度，以及支路铺设的深度。把同样的参数预算换成通用的低秩适配器，收益接近谱方案的三倍。而在需要真正检索远方证据的任务上，无论怎么配置，修正都交不出答卷——它能做的是在窗口边缘附近校准模型的"语气"，而不是把远方的具体事实找回来。效率方面，默认配置让模型读入长文的阶段慢 2 至 3 倍，逐词生成的阶段则与未修正的模型持平。

本文的贡献因此不是又一种高效注意力变体，而是一幅边界图：一个预算极小的谱修正，能补回什么、不能补回什么、收益由什么构成、代价花在哪里。全部数字都锚定在同一套可重复执行的评估协议上，每个结论都可以被独立复核。对正在设计局部长上下文模型的实践者，这幅图可以直接当作决策参照。

## 2. 相关工作

滑动窗口注意力的思想在一系列结构化稀疏注意力工作中奠定基础。Longformer 与 BigBird 让少数位置看得更远，从而把注意力的成本控制在可接受范围；Mistral 则证明滑动窗口足以作为生产级模型的默认配置。这些方法在训练之初就把稀疏结构写进模型，与模型权重一同长大；我们研究的是一个互补的问题——模型已经定型之后，还能补装什么。

另一条路线用核函数近似标准注意力。线性注意力与 Performer 用特征映射把"查询与键的相似度"改写成可以预先汇总的形式，从而绕开平方复杂度。Hedgehog 一文的分析与我们尤为相关：它指出 ELU+1 这类正特征映射天然缺少标准注意力的尖锐性，难以表达"精确选中某一个位置"。我们远支路使用的恰是这类映射，而我们的实证结果——统计平滑有效、精确检索无效——为这一理论预期提供了直接的对应证据。

把局部与全局通路拼装起来的混合架构与我们的出发点更接近。BASED 把精确局部注意力与一条全局线性分支结合，Infini-attention 用压缩记忆扩展注意力的视野。两者都在训练时端到端地塑造全局通路；MSRA 则冻结基座、只嫁接一条极小的残差支路，回答的是一个诊断性问题：一条全局通路在多大程度上可以事后补装？我们的答案——对整体语言建模质量可以、对精确检索不可以——为这类设计提供了参照点。

最后，参数高效适配领域早已证明，几万个参数足以显著改变一个冻结模型的行为，LoRA 是其中影响最广的代表。我们把谱修正与同预算的 LoRA、以及一种朴素的深度卷积放在完全相同的数据、训练与评估协议下比较。据我们所知，这种"专门结构对通用适配器"的同协议对照，在长上下文修正的文献里尚不多见，而正是这种对照给出了本文最有分量的一个结论。

## 3. 方法：多尺度谱残差注意力

### 3.1 直觉

想象每个词是会议室里的一名听众。它面前摊着笔记本，详细记下最近的发言——这是窗口内的精确注意力；至于更早的发言，它只剩一片模糊印象。MSRA 的想法，是给坐在后排的听众配一副"远视眼镜"：把窗外所有历史按距离远近加权，越远权重越小，汇总成一份补充摘要，夹在笔记本里一并参考。眼镜本身的结构极其简单——每层只有约两千三百个零件——而且听众原本的笔记方式一个字也不改。

### 3.2 定义

模型基座完全冻结。对最后 L 层的注意力输出，在精确的局部结果之外叠加一份远场残差：

$$y_i = y_i^{near} + \gamma \cdot y_i^{far}$$

这里 y_i^near 是滑动窗口注意力的原始输出，y_i^far 是远支路的输出，γ 是每层一个的可学习增益，初始化接近零，让模型从"纯局部"的状态平滑起步。

远支路对每个 expert（可以理解为一种"视野尺度"）m 计算两组加权和：

$$C_i^{(m)} = \sum_{j \in F(i)} g_m(i-j)\,\psi_m(k_j)\,v_j^\top,\quad z_i^{(m)} = \sum_{j \in F(i)} g_m(i-j)\,\psi_m(k_j)$$

$$u_i^{(m)} = \frac{\varphi(q_i)^\top C_i^{(m)}}{\varphi(q_i)^\top z_i^{(m)} + \varepsilon},\qquad y_i^{far} = \sum_m \lambda_{im}\, u_i^{(m)}$$

其中 F(i) 是位置 i 的远场证据集，即窗口之外、历史方向上的全部位置；g_m(Δ)=exp(−Δ/σ_m) 是指数衰减核，距离越远、权重越小，衰减的快慢由尺度 σ_m 控制；φ 与 ψ 是 ELU+1 正特征映射，保证分母恒正、训练稳定；λ 是各视野尺度的混合权重，可以由一个逐词的门控网络产生（gate-on），也可以简单地取均匀平均（gate-off）。直观地说，这组公式做的事情就是：把窗口外的历史按距离衰减后汇总，再用当前词的查询向量做一次温和的匹配读取。

### 3.3 设计约定

为了让归因干净，我们把可变的部分压缩到最少。核的形状固定为指数衰减，只有衰减尺度 σ 可以学习；基座权重一律冻结；可训练参数只存在于修正支路内部——特征映射、核尺度、可选的门控，以及每层的输出增益 γ。使用 M 个 expert 时，每个 expert 拥有独立的特征映射与尺度，初始尺度按对数均匀排布（对应约 20 到 400 个词元的衰减长度），构成一组从近到远的"多尺度视野"。

### 3.4 高效实现

指数核只依赖距离差，因此上式中的两组求和都是卷积，可以用快速傅里叶变换在 O(n log n) 内完成；推理时按尺度缓存核的变换结果，不必重复计算。

模型的工作分为两个阶段：读入提示词的阶段（预填充）会一次性处理整段文本，远支路只在这个阶段计算；此后逐词生成的阶段（解码），模型退回纯局部注意力，通过滑动窗口 KV 缓存工作——可以把缓存想成一块随生成不断追加的速记板，每个新词只需翻阅板上最近的 w+1 条记录，而不必重读全文。远支路在解码阶段结构性无法激活，因此解码的速度与未经修正的局部注意力完全相同（见表 4）。

### 3.5 默认配置

除非特别说明，默认配置为单尺度（M=1、无门控）、嫁接最后 8 层、窗口 w=128、特征秩 r=16，每层 2,306 个可训练参数，在 LLaMA 3.2-1B 上合计 18,448 个。GPT2-XL 实验使用相同配置。支路内全部线性投影零初始化，保证训练起点严格等于未修正的模型。

## 4. 实验设计

### 4.1 数据与评估协议

语言建模实验在 PG-19 长篇小说语料上进行，训练按文档 75/25 切分。评估集是一张一次性固定的 16 段文本清单，其抽取规则、入选书目与每册词元数都记录在一个带 SHA-256 校验的清单文件里——每个待测配置回答的是同一份考卷，横向比较因此有意义。所有配置另在 PG-19 官方测试分区（训练从未接触的部分）复核，结论一致。

主指标是差距闭合率。局部注意力与全局注意力的困惑度之差，就是窗口丢掉的全部信息所值的价格；修正补回了其中的多大比例，就记为多少。以 LLaMA @512 为例：局部注意力困惑度 371.6，全局 17.0，MSRA 修正后 327.1——闭合率即 (371.6−327.1)/(371.6−17.0) ≈ 12.4%。困惑度本身可以理解为模型对下一个词的"惊讶程度"，越低越好。

检索能力用经典的"大海捞针"测试度量：把一句"The secret key is XYZZY-789"这样的合成句子藏进一段长篇小说，在文末提问，看模型能否复述密钥。我们按证据到问题的距离分三桶——窗口内、刚出窗口（一倍到两倍窗口）、远处（两倍窗口以外）——测量模型在答案词元上的惊讶程度。协议有两条硬性纪律：藏针先于裁剪，且每枚针在裁剪后逐例断言完整存活（聚合存活率为 1.0）；损失严格落在答案词元上。另有两项如实披露：距离最近的样本中，针会略向左移以保证完整放下，实际距离因此略大于名义距离但仍处于窗口内；问题位置的估算会给桶边界带来约 6 个词元的滑差。

任务层面的检验使用 LongBench 官方指标（问答 F1 与摘要 ROUGE-L），贪心解码，三个被比较的对象共享完全相同的提示词（左侧裁至 2,048 词元）与生成设置。RULER 的 needle-aware 复测作为补充对照。效率测量在同一台机器、同一时段内配对进行，覆盖 512 至 4,096 词元的读入吞吐与逐词生成吞吐。

训练协议对全部配置统一：AdamW 优化器（谱参数学习率 2e-4；适配器基线按其惯例取更保守的 1e-4），余弦退火，3,000 步，文档级采样。每个训练结构至少跑 2 个随机种子，主配置 3 至 5 个；只含增益标量的探针配置为 1 个种子。

### 4.2 比较对象

要判断一条修正支路的贡献，需要一组层层递进的对照。最外侧的两根门柱是全局注意力与局部注意力：前者看得见全文，定义了差距的下界；后者是未修正的基座，定义了起点。谱家族内部，我们安排了一条从简到繁的阶梯：最简的是"仅增益"探针，只训练每层的输出强度标量（8 层配置为 8 个标量，16 层为 16 个），其余全部冻结；往上依次是单尺度家族加深度的三档（M1L4、M1L8、M1L16），以及多尺度家族（M4L4、M4L8，各配一个无门控对照）。谱家族之外有两个"通用对手"：与支路同预算的 rank-1 LoRA 适配器，和一种朴素的因果深度可分离卷积——它们不含任何谱结构，用来回答"同样的参数预算，不讲究结构会怎样"。最后还有一个安慰剂对照：一条完全未经训练、随机初始化的支路，用来验证收益确实来自学习而非注入本身。全部配置与参数量列于下表。

| 类别 | 配置 | 可训练参数 |
|---|---|---|
| 基线 | 全局注意力 / 局部注意力（w=128） | 0 |
| 谱家族 | 仅增益探针（L8 / L16，每层一个标量，与尺度数无关） | 8–16 |
| | M1L4 / M1L8 / M1L16（单尺度，深度扫描） | 9.2K / 18.4K / 36.9K |
| | M4L4 / M4L8（多尺度，各带无门控对照） | 21.5–45.1K |
| 通用适配器 | rank-1 LoRA（作用于末 8 层输出投影） | 32.8K |
| | 因果深度可分离卷积（k=8，作用于输出投影） | 131.1K |
| 安慰剂对照 | 未训练随机核 | 0 |

## 5. 实验结果

### 5.1 语言建模：增益真实且稳定

**表 1：PG-19 困惑度（固定 16 段评估集）**

| 模型 | 训练长度 | 全局 | 局部 | MSRA (M1L8) | 差距闭合 |
|---|---|---|---|---|---|
| GPT2-XL (1.5B) | 512 | 42.6 | 2,649±571 | 2,373±535 | **10.6%** |
| LLaMA 3.2-1B | 512 | 17.0 | 371.6 | 327.1 | **12.4%** |
| LLaMA 3.2-1B | 1024 | 18.1 | 545.7 | 474.3 | **13.5%** |

三个事实值得先看清楚。局部注意力在小说语料上的退化极为严重——LLaMA @512 的困惑度从 17 涨到 372，这不是百分之几的损失，而是二十多倍的劣化，说明窗口丢弃的信息对长篇小说几乎不可或缺。修正的效果稳定为正：各配置的平均增益落在 10.4% 至 13.1% 之间（GPT2-XL 为 10.4%，LLaMA @512 为 12.0%），且每一个种子都为正——GPT2-XL 的五个逐种子增益分布在 9.8% 至 10.9%，无一为负。把评估集整体换成官方测试分区，LLaMA 上的增益依然成立（+10.1%），排除了"恰好选对考卷"的可能。

换算成预算语言：18,448 个参数约为基座的百万分之十五（0.0015%），换得约 12% 的困惑度下降。这个交换比乍看相当划算；它值不值得做，取决于这 12% 是什么性质——这正是接下来三节要拆解的问题。

### 5.2 归因阶梯：增益从哪里来

**表 2：归因阶梯（全部独立进程评测、同一固定评估集）**

| 修正 | 参数 | 差距闭合 | 种子数 |
|---|---|---|---|
| 未训练随机核 | 18.4K | 0.0% | — |
| 仅增益（M1L8） | 8 | 10.0% | 1 |
| 仅增益（M4L8） | 8 | 10.1% | 1 |
| 仅增益（M1L16） | 16 | 26.4% | 1 |
| M1L4（无门控） | 9.2K | 5.3–5.4% | 2 |
| M4L4（无门控） | 21.5K | 6.1% | 2 |
| M1L8（默认配置） | 18.4K | 12.4% | 3 |
| M4L4（门控） | 22.5K | 8.6–8.7% | 2 |
| M4L8（无门控） | 43.0K | 15.4% | 2 |
| M4L8（门控） | 45.1K | 25.7% | 2 |
| M1L16（无门控） | 36.9K | **31.7%** | 2 |
| 深度可分离卷积 k=8 | 131.1K | 78.3% | 2 |
| rank-1 LoRA | 32.8K | **87.4%** | 2 |

这张表的读法是从上往下逐级加注：每一行相对上面的行，要么多训练一个零件，要么加深一层嫁接。安慰剂对照率先给出定心丸——随机初始化的支路闭合率为零，说明收益全部来自学习。

第一个发现是，收益的绝大部分来自一个出人意料的零件：每层的输出增益标量。只训练这 8 个标量（M1L8 骨架），闭合率已达 10.0%；换成多尺度骨架但同样只训练 8 个标量，结果几乎一样（10.1%）。把深度加到 16 层，仅 16 个标量就闭合 26.4%。换句话说，在两种深度下，纯增益都贡献了完整结构收益的 81% 至 83%，完整的谱结构（特征映射、核尺度、门控）在其上只追加 2.4 至 5.4 个百分点。更值得玩味的是增益通道自身的缩放行为：深度翻倍，闭合率从 10.0% 涨到 26.4%，远超线性——我们在 5.3 节会看到，这条通道的本质是对输出的校准。

第二个发现是深度的主导地位。单尺度家族沿深度单调上升且斜率变大：4 层 5.4%、8 层 12.4%、16 层 31.7%。把比较换成预算匹配的形式，结论更加锋利——多尺度 M4L8 的预算（45.1K 参数，25.7%）改投深度得到的 M1L16，参数更少（36.9K）而闭合更高（31.7%）；M4L4 门控版的预算（22.5K，8.6%）改投深度得到的 M1L8，同样参数更少（18.4K）而闭合更高（12.4%）。诚实的边界也需要写明：在 4 层处，原始的跨格比较方向相反（门控多尺度的 8.6% 高于单尺度的 5.4%），这正是我们以预算匹配而非逐点形式陈述此结论的原因。

第三个发现关于多尺度与门控：贡献真实，但居于次要。无门控时，多尺度在 4 层和 8 层分别追加 0.7 与 3.0 个百分点；门控的贡献随深度放大，从 4 层的 2.5 个百分点增至 8 层的 10.3 个百分点。训练后的门控确实学到了选择性——平均门控熵只有均匀分布的 62.8%，且扰动查询词对门控的撼动是扰动远端上下文的 113 倍。多尺度配置的核尺度在训练后相对初始化温和漂移（−10% 至 +14%），说明初始化的尺度排布基本合理，训练起的是精炼作用；单尺度配置中增益之外的轨迹更小（详见补充材料）。

第四个发现是分量最重的。同预算的 rank-1 LoRA——一种不含任何谱结构、只是在注意力输出投影旁并联低秩矩阵的通用适配器——以 32.8K 参数闭合 87.4%，是谱家族最强配置的 2.8 倍；即便把七倍预算花在一种平凡的深度卷积上，也能达到 78.3%。两个通用适配器的前后夹击说明：局部与全局之间的差距，主体是通用容量缺口，而非只有谱结构才能表达的远方信息缺口——谱形式在这项任务上并不构成必要的归纳偏置。

不过有一个生态位值得点名。当预算压缩到四千参数以下，通用适配器连实例化都做不到——rank-1 LoRA 哪怕只作用在一层的输出投影上也需要 4,096 个参数——而 16 个增益标量闭合了 26.4%。在超低预算区间，谱式增益校准没有通用对手，未来的匹配预算适配研究都应把它列为对照。

### 5.3 增益的性质：平滑而非检索

困惑度只能说明整体语言建模质量变好，不能说明模型找回了远方的具体事实。大海捞针测试把答案放在不同距离上，给出的图景要微妙得多。

**表 3：NIAH 答案词元 NLL（w=128，2,048 词元；33 枚针、每桶 11 枚，逐例断言存活、聚合率 1.0；± 为跨样本标准差；括号为相对局部–全局差距的闭合率，负值表示修正后比局部更差）**

| 配置 | 窗口内 (≤128) | 刚出窗口 (129–256) | 远处 (>256) |
|---|---|---|---|
| 全局 | 0.82±0.15 | 0.80±0.17 | 0.91±0.11 |
| 局部 | 2.37±0.83 | 9.35±1.79 | 9.75±0.25 |
| + M1L8 | 2.60±0.70 (−15.2%) | 8.62±2.49 (+8.6%) | 9.72±0.17 (+0.4%) |
| + M4L4 | 2.57±0.70 (−12.8%) | 8.71±2.58 (+7.5%) | 9.75±0.18 (+0.0%) |
| + M4L8 | 2.49±0.66 (−7.6%) | 8.64±2.58 (+8.4%) | 9.79±0.16 (−0.4%) |
| + M1L16 | 2.43±0.54 (−4.3%) | 8.73±2.40 (+7.3%) | 9.85±0.16 (−1.1%) |
| + 仅增益（M1L8） | 2.59±0.70 (−14.3%) | 8.61±2.50 (+8.7%) | 9.69±0.17 (+0.7%) |
| + 仅增益（M1L16） | 2.42±0.53 (−3.5%) | 8.66±2.35 (+8.2%) | 9.68±0.16 (+0.8%) |

先看窗口之内。证据明明就在眼前，局部注意力本来做得不错（2.37 对全局的 0.82），六种修正配置却一致地让它变得略差（−3.5% 至 −15.2%）。原因在于注入本身：远支路把窗口外历史的平滑平均叠加进来，稀释了局部注意原本尖锐的聚焦。代价随深度收窄——M1L8 为 −15.2%，M1L16 仅 −4.3%，我们将其作为描述性事实报告，机制尚无可验证的解释。尤其说明问题的是，仅增益探针的代价与完整结构几乎相同（8 层：−14.3% 对 −15.2%；16 层：−3.5% 对 −4.3%）——代价跟着增益通道走，与核结构无关。

再看刚出窗口的一小段。这里是局部注意力的盲区边缘，指数核的衰减权重又尚未归零，六个配置的答案 NLL 一致改善局部–全局差距的 7% 至 9%。这是整条支路唯一真正帮到检索类预测的地带，机制同样经由增益通道——仅增益探针的签名与完整结构一致。

远处则是一片空白。超过两倍窗口，没有任何配置的改善超过 0.8%，最深的配置反而差 1.1%。有一个看似例外的数据点需要直面：在 1,024 词元下，M1L16 的远桶闭合 +15.0%（NLL 从 10.66 降至 9.21）。但这两个数值仍比全局基线高出一个数量级，且该效应在 2,048 词元下完全消失（−1.1%）——我们判定它是对远距通用文本的先验校准改善，而非证据检索。这与 5.2 节的归因互为印证：谱修正补回的是"窗口外信息的平滑平均"，这类信号足以温和地调整模型的预期，却承载不起对某个具体远方事实的精确提取——后者需要尖锐的、按内容选择的注意力，而这恰是固定指数核与正特征映射在设计上不具备的能力。

任务指标给出了同一结论的另一半。LongBench 官方指标下，三个被比较对象都处在指标地板附近：全局注意力均值 0.080，局部 0.033，修正后 0.033。窗口的真实代价精确地落在需要长程整合的任务上——gov_report 的 ROUGE-L 从 0.137 跌到 0.003，triviaqa 的 F1 从 0.197 跌到 0.0，且失败的形态是生成重复退化的文本（如 "Answer: Answer: …"）而非渐进劣化。其余五个任务上三者生成逐字相同的地板级文本，闭合率在该处无定义；在有差异的三个任务上，修正的闭合率至多 1.3%。RULER 的 needle-aware 复测（全局臂作为阳性对照，困惑度 1.56，确认协议本身可检出检索能力）同样只给出 3.0% 的闭合。困惑度的收益，不转化为任务能力。

### 5.4 效率

**表 4：预填充吞吐减速（每行为同卡同 session、以即时重测的局部注意力为锚的配对测量）**

| 配置 | @512 (tok/s) | @1,024 | @2,048 | @4,096 | 解码 @2,048 |
|---|---|---|---|---|---|
| + M1L8（默认） | 2.9× (12,790) | 2.5× (12,020) | 2.0× (9,961) | 显存超限 | ≈1.0× (115.4 vs 115.2) |
| + M1L16 | 4.8× (7,641) | 4.1× (7,499) | 3.0× (6,640) | 显存超限 | ≈1.0× (114.8 vs 115.1) |
| + M4L8 | 8.7× (4,541) | 6.7× (4,388) | 5.0× (4,046) | 显存超限 | ≈1.0× (114.0 vs 114.8) |

三行各自的同 session 局部锚（@512）为 36,956 / 36,991 / 39,532 tok/s；减速比率跨 session 复现于约 6% 以内。读表可以得到三件事。减速随序列变长而摊薄——局部注意力本身的成本随长度增长，远支路的 O(n log n) 开销占比随之下降；减速随远支路的工作量（嫁接层数 × 尺度数：8、16、32 个单元）单调放大，@512 处为 2.9×、4.8×、8.7×；瓶颈是 FP32 FFT 中间张量的显存流量，@2,048 峰值显存从局部的 3.4GB 涨到 M1L8 的 13.1GB、M1L16 的 20.2GB，4,096 词元以上全部谱配置超出 24GB 显存（局部臂本身仍能以 12,457 tok/s 运行）。

解码则是另一个故事。远支路在解码期结构性关闭，修正模型每生成一个词的计算量按构造与局部完全相同；两个 session 的六次配对测量落在 0.99 至 1.10 之间，中位数 1.004——统计上与局部不可区分。表 4 解码列印出的是干净 session 的配对值。

## 6. 结论与展望

回到最初的问题：一个冻结的局部注意力模型，用几万个参数的事后修正，能补回多少窗口外的信息？本文给出的完整答案是——能补回真实而稳定的一部分（最多闭合 31.7% 的困惑度差距），但这部分收益的构成与性质都值得细看。收益由学习到的输出增益与嫁接深度主导（仅 16 个增益标量即可闭合 26.4%，完整结构为 31.7%），多尺度核与门控提供真实但次要的贡献；增益的本质是 token 似然的统计平滑——它改善刚出窗口处的预测，轻微牺牲窗口内的精确性，对远距离检索无能为力，也不迁移到下游任务指标。而在同预算对照下，通用低秩适配以 87.4% 对 31.7% 的显著优势表明：这个差距的主体是通用容量，谱结构不是必需的归纳偏置。

对实践者，建议因此相当直接。目标是降低冻结模型的困惑度，选 LoRA，更简单也更强；坚持谱形式，就把预算投给嫁接深度而非核的多样性；任何此类修正都不应被期待恢复远距离检索。唯一例外是超低预算生态位——四千参数以下通用适配器无法实例化，而增益标量在那里没有竞争者。

对更长远的研究，本文的负结果同样指路。远场检索的失败是结构性的：固定核与正特征映射表达不了内容选择性。这提示事后补装的全局通路若要获得检索能力，可能需要可寻址的记忆形式——压缩的 KV 或显式检索——而非平滑的谱聚合。与此同时，修正唯一改善 NLL 的地带恰在窗口边缘、局部失明之处，说明"在窗口边缘校准表示"是低预算适配中被低估的收益来源。

本文的限制需要如实列出。实验规模为 1B–1.5B 模型与不超过 2,048 词元的训练长度，向更大模型或更长上下文外推需谨慎；检索与任务层面的分析（NIAH、LongBench、RULER）只在 LLaMA 3.2-1B 上完成，GPT2-XL 仅用于验证困惑度增益的架构泛化；评估窗口单一（w=128），窗口与核尺度的联合扫描留待后续；LongBench 覆盖 15 个英文任务中的 7 个，三方协议完全匹配，结论预期不因任务集扩大而改变；多尺度在 16 层的配置未测试，但 LoRA 87.4% 对家族最优 31.7% 的支配性结论不依赖它；谱支路的预填充开销未做内核级优化，实测减速是未优化实现的上界而非方法本身的下界。实现层的全部调试记录、九环修正总表与机器断言清单收录于补充材料。

---
---


# MSRA: A Parameter-Efficient Spectral Correction for Local Attention — Where It Helps, and Where It Fails

*A Systematic Attribution Study of a Lightweight Spectral Correction*

---

## Abstract

Long documents pose an expensive dilemma for large language models. Letting every token attend to the entire text costs compute that grows quadratically with length; restricting attention to a sliding window brings the cost down to linear, but everything beyond the window is forgotten outright. This paper asks a direct question: if the model is already trained, its original weights are frozen, and the budget for new parameters is only a few tens of thousands, can we still recover what was lost — and what exactly would we recover?

We propose Multi-scale Spectral Residual Attention (MSRA): a far-field branch grafted onto the last few layers of a frozen model, which condenses all out-of-window history into a supplementary summary through smoothly distance-decayed weights, and adds it back to the attention output. The branch carries roughly nine to forty-five thousand trainable parameters, with an additional set of 8-to-16-scalar minimal probes for attribution. On PG-19 long-form fiction, the branch closes up to 31.7% of the perplexity gap between local and full attention (LLaMA 3.2-1B, sixteen-layer configuration), with gains stable across two architectures and multiple random seeds, and replicated on the official test partition.

The more interesting findings lie in the attribution. The gains come from two unexpectedly plain sources: a single learnable output-strength scalar per layer — training only these 16 scalars already closes 26.4% — and the depth over which the branch is spread, whose returns grow monotonically and accelerate. The carefully designed kernel structure and gating contribute only secondarily, while a budget-matched generic adapter (LoRA) closes 87.4%, indicating that the gap is mostly a generic capacity deficit rather than a missing far-field pathway. Retrieval experiments reveal the correction's true nature: it is a statistical smoother that improves prediction just beyond the window, slightly degrades precise in-window answer prediction, fails entirely at genuine long-distance retrieval, and does not transfer to long-document task metrics. In efficiency, the default configuration slows prefill by two to three times while decoding is entirely unaffected. Together, these results provide a complete, reproducible map of what a lightweight spectral correction can and cannot recover, offering direct empirical guidance for the design of local long-context models.

---

## 1. Introduction

One of the mainstream ways modern language models handle long sequences is to let each token attend only to the most recent w tokens. This design compresses the cost of attention from quadratic to linear in sequence length, making documents of thousands of words practical to process. Yet anyone who has read a novel knows how much a hint planted in chapter three can matter — and to a windowed model, chapter three might as well not exist.

A large body of work tries to close this gap. Sparse attention lets selected positions see farther, linear attention replaces the standard mechanism with kernel approximations, state-space models take a different route to memory altogether, and hybrid local–global architectures add a global pathway alongside the window. Most of these solutions replace the attention mechanism itself, or require retraining the model end to end. Some settings afford no such luxury — the model is already deployed, its original weights are untouchable, and the budget for new parameters is a few tens of thousands. Our question follows: under constraints this tight, how much of the lost information can a spectral far-field correction actually recover? And what is the recovered portion made of?

We propose MSRA: grafting a frequency-domain residual branch onto the last L layers of a frozen model, training only that branch. The setting has clear practical relevance — deployed models that cannot be retrained, extremely low adaptation budgets — and it doubles as a clean attribution workbench, because every component of the branch (output gain, kernel structure, multi-scale, gating, grafting depth) can be switched on or off independently and examined one by one.

With this workbench we measured thirteen configurations systematically, and the picture that emerged is sharper than expected. The perplexity improvement is real and stable across seeds, yet its source is not the spectral structure we designed but two plainer factors: a per-layer learnable output strength, and the depth over which the branch is spread. Spending the same parameter budget on a generic low-rank adapter yields nearly three times the recovery. And on tasks that require retrieving specific distant evidence, the correction has nothing to offer — what it can do is calibrate the model's expectations near the window's edge, not fetch facts from afar. On efficiency, the default configuration slows the reading-in stage by two to three times, while word-by-word generation runs at exactly the speed of the uncorrected model.

The contribution of this paper is therefore not another efficient attention variant, but a boundary map: what a minimally-budgeted spectral correction recovers, what it cannot, what its gains are made of, and where its costs lie. Every number is anchored to a single reproducible evaluation protocol, and every conclusion can be independently re-checked. For practitioners designing local long-context models, this map is meant to serve directly as a decision reference.

## 2. Related Work

The idea of sliding-window attention builds on a line of structured sparse attention. Longformer and BigBird let selected positions see farther, keeping attention affordable; Mistral showed that a sliding window suffices as a production default. These methods bake the sparse structure into the model from the start of training, letting it grow together with the weights; we study the complementary question of what can still be retrofitted once the model is fixed.

A second line approximates standard attention with kernels. Linear attention and Performer rewrite query–key similarity into a form that can be aggregated in advance, sidestepping quadratic cost. The analysis of Hedgehog is especially relevant to us: it shows that positive feature maps of the ELU+1 family inherently lack the sharpness of softmax attention and struggle to single out one exact position. Our far branch uses precisely such a map, and our empirical findings — statistical smoothing works, precise retrieval does not — provide direct evidence for that theoretical expectation.

Hybrid architectures that combine local and global pathways sit closest to our starting point. BASED couples exact local attention with a global linear branch; Infini-attention extends attention with compressive memory. Both train the global pathway end to end; MSRA instead freezes the base and grafts a minimal residual, answering a diagnostic question: to what extent can a global pathway be retrofitted after the fact? Our answer — yes for overall language-modeling quality, no for precise retrieval — offers a reference point for such designs.

Finally, the parameter-efficient adaptation literature has long shown that tens of thousands of parameters can substantially change a frozen model's behavior, with LoRA the most influential example. We compare the spectral correction against same-budget LoRA, and against a plain depthwise convolution, under identical data, training, and evaluation protocols. To our knowledge, such same-protocol comparisons of "specialized structure versus generic adapter" are rare in the long-context correction literature — and it is exactly this comparison that yields one of this paper's weightiest conclusions.

## 3. Method: Multi-scale Spectral Residual Attention

### 3.1 Intuition

Picture each token as a listener in a meeting room. It keeps a notebook with detailed records of the most recent statements — that is the exact attention within the window; of everything said earlier, only a vague impression remains. MSRA's idea is to hand the listeners in the back row a pair of binoculars: everything beyond the window is weighted by distance, farther statements weighted less, and condensed into a supplementary summary tucked into the notebook. The binoculars themselves are extremely simple — about twenty-three hundred parts per layer — and the listeners' original note-taking is left untouched.

### 3.2 Definition

The base model is entirely frozen. On the attention output of the last L layers, a far-field residual is added to the exact local result:

$$y_i = y_i^{near} + \gamma \cdot y_i^{far}$$

Here y_i^near is the original output of sliding-window attention, y_i^far is the output of the far branch, and γ is a per-layer learnable gain initialized near zero, so the model starts smoothly from a purely local state.

For each expert m — think of it as one "field of view" — the far branch computes two weighted sums:

$$C_i^{(m)} = \sum_{j \in F(i)} g_m(i-j)\,\psi_m(k_j)\,v_j^\top,\quad z_i^{(m)} = \sum_{j \in F(i)} g_m(i-j)\,\psi_m(k_j)$$

$$u_i^{(m)} = \frac{\varphi(q_i)^\top C_i^{(m)}}{\varphi(q_i)^\top z_i^{(m)} + \varepsilon},\qquad y_i^{far} = \sum_m \lambda_{im}\, u_i^{(m)}$$

F(i) is the far-field evidence set of position i — every position beyond the window, in the past direction; g_m(Δ)=exp(−Δ/σ_m) is an exponential decay kernel whose falloff speed is set by the scale σ_m; φ and ψ are ELU+1 positive feature maps, which keep the denominator positive and training stable; λ are the mixing weights across fields of view, produced either by a per-token gating network (gate-on) or by simple uniform averaging (gate-off). In plain terms, these equations gather out-of-window history with distance-decayed weights, then let the current token's query read from that aggregate in a gentle, normalized way.

### 3.3 Design Conventions

To keep attribution clean, we compress the movable parts to a minimum. The kernel shape is fixed as exponential decay, with only its scale σ learnable; the base weights are all frozen; trainable parameters live only inside the correction branch — the feature maps, the kernel scales, the optional gate, and the per-layer output gain γ. With M experts, each carries its own feature map and scale, initialized at log-uniform spacings (decay lengths of roughly 20 to 400 tokens), forming a set of multi-scale views from near to far.

### 3.4 Efficient Implementation

Because the exponential kernel depends only on distance, both sums above are convolutions and can be computed in O(n log n) with the fast Fourier transform; at inference the transformed kernels are cached by scale rather than recomputed.

A model works in two stages. The reading-in stage (prefill) processes the whole prompt at once, and the far branch computes only there; afterwards, the word-by-word generation stage (decoding) falls back to pure local attention through a sliding-window KV cache — picture the cache as a notepad that grows as text is generated, with each new word consulting only the most recent w+1 entries instead of rereading the document. The far branch is structurally unable to fire during decoding, so generation runs at exactly the speed of uncorrected local attention (Table 4).

### 3.5 Default Configuration

Unless stated otherwise, the default configuration is single-scale (M=1, no gating), grafted on the last 8 layers, window w=128, feature rank r=16: 2,306 trainable parameters per layer, 18,448 in total on LLaMA 3.2-1B. GPT2-XL experiments use the same configuration. All linear projections in the branch are zero-initialized, so the starting point of training is exactly the uncorrected model.

## 4. Experimental Design

### 4.1 Data and Evaluation Protocol

Language-modeling experiments run on PG-19 long-form fiction, with documents split 75/25 for training. The evaluation set is a one-time fixed list of 16 text chunks whose sampling rule, selected books, and per-book token counts are recorded in a manifest file with a SHA-256 checksum — every configuration answers the same exam paper, which is what makes the comparisons meaningful. All configurations are additionally verified on the official PG-19 test partition, which training never touches, with consistent conclusions.

The primary metric is gap closed. The perplexity difference between local and full attention is the price of everything the window throws away; the fraction of that difference a correction recovers is its score. Concretely, for LLaMA at 512 tokens: local attention sits at 371.6, full attention at 17.0, and MSRA at 327.1 — a closure of (371.6−327.1)/(371.6−17.0) ≈ 12.4%. Perplexity itself can be read as the model's average "surprise" at the next word; lower is better.

Retrieval is measured with the classic needle-in-a-haystack test: hide a synthetic sentence such as "The secret key is XYZZY-789" inside a long novel, ask about it at the end, and see whether the model can reproduce the key. We bucket examples by the distance from evidence to question — inside the window, just outside it (one to two windows away), and far (beyond two windows) — and measure the model's surprise on the answer tokens. Two protocol rules are enforced at runtime: insertion happens before any truncation, and every needle is asserted to survive truncation intact (the aggregate survival rate is 1.0); the loss falls strictly on answer tokens. Two further disclosures are made plainly: at the smallest intended distances the needle shifts slightly leftward to fit, so its actual distance exceeds the nominal one while remaining inside the window; and estimating the query position introduces a boundary slide of about six tokens between buckets.

Task-level checks use official LongBench metrics (F1 for question answering, ROUGE-L for summarization) under greedy decoding, with the three compared models sharing identical prompts (left-truncated to 2,048 tokens) and generation settings. A needle-aware RULER retest serves as a complementary control. Efficiency is measured pairwise on the same machine within the same session, covering reading throughput from 512 to 4,096 tokens and generation throughput.

The training protocol is uniform across configurations: AdamW (learning rate 2e-4 for spectral parameters; the adapter baselines follow their convention at the more conservative 1e-4), cosine annealing, 3,000 steps, document-level sampling. Every trained structure runs at least 2 random seeds, primary configurations 3 to 5; the gain-only probes run 1 seed.

### 4.2 What We Compare Against

Judging a correction branch requires a ladder of controls. The two outermost goalposts are full attention and local attention: the former sees the whole text and defines the bottom of the gap; the latter is the uncorrected base and defines the starting line. Within the spectral family, the ladder runs from simple to elaborate. At the bottom sit the gain-only probes, which train nothing but the per-layer output-strength scalars (8 scalars for the 8-layer skeleton, 16 for the 16-layer one) with everything else frozen. Above them are the single-scale family at three depths (M1L4, M1L8, M1L16) and the multi-scale family (M4L4, M4L8), each multi-scale row paired with a gate-off control. Outside the spectral family stand two generic opponents: a budget-matched rank-1 LoRA adapter and a plain causal depthwise convolution — neither contains any spectral structure, and they answer the question "what happens with the same budget but no structural priors?" Last comes a placebo control: a randomly initialized, never-trained branch, verifying that the gains come from learning rather than from injection itself. All configurations and their parameter counts appear in the table below.

| Category | Configuration | Trainable params |
|---|---|---|
| Baselines | Full attention / Local attention (w=128) | 0 |
| Spectral family | Gain-only probes (L8 / L16; one scalar per layer, independent of the number of scales) | 8–16 |
| | M1L4 / M1L8 / M1L16 (single-scale depth sweep) | 9.2K / 18.4K / 36.9K |
| | M4L4 / M4L8 (multi-scale, each with a gate-off control) | 21.5–45.1K |
| Generic adapters | rank-1 LoRA (on the output projections of the last 8 layers) | 32.8K |
| | Causal depthwise separable convolution (k=8, on output projections) | 131.1K |
| Placebo control | Untrained random kernel | 0 |

## 5. Results

### 5.1 Language Modeling: Gains Are Real and Stable

**Table 1: PG-19 perplexity (fixed 16-chunk evaluation set)**

| Model | Train length | Full | Local | MSRA (M1L8) | Gap closed |
|---|---|---|---|---|---|
| GPT2-XL (1.5B) | 512 | 42.6 | 2,649±571 | 2,373±535 | **10.6%** |
| LLaMA 3.2-1B | 512 | 17.0 | 371.6 | 327.1 | **12.4%** |
| LLaMA 3.2-1B | 1024 | 18.1 | 545.7 | 474.3 | **13.5%** |

Three facts deserve attention first. The degradation of local attention on fiction is severe — LLaMA's perplexity at 512 tokens jumps from 17 to 372, not a loss of a few percent but a twenty-fold worsening, which says the information outside the window is nearly indispensable for long novels. The correction's effect is stably positive: configuration-mean gains span 10.4% to 13.1% (10.4% for GPT2-XL, 12.0% for LLaMA at 512), and every individual seed is positive — GPT2-XL's five per-seed gains range from 9.8% to 10.9%, none negative. Replacing the evaluation set wholesale with the official test partition leaves the conclusion standing (+10.1% on LLaMA), ruling out a lucky choice of exam paper.

In budget terms: 18,448 parameters — about fifteen millionths of the base (0.0015%) — buy roughly a 12% perplexity reduction. Whether that trade is worth making depends on what the 12% is made of, which is exactly what the next three sections take apart.

### 5.2 The Attribution Ladder: Where the Gains Come From

**Table 2: Attribution ladder (all independent-process evaluations on the same fixed evaluation set)**

| Correction | Params | Gap closed | Seeds |
|---|---|---|---|
| Untrained random kernel | 18.4K | 0.0% | — |
| Gain-only (M1L8) | 8 | 10.0% | 1 |
| Gain-only (M4L8) | 8 | 10.1% | 1 |
| Gain-only (M1L16) | 16 | 26.4% | 1 |
| M1L4 (gate-off) | 9.2K | 5.3–5.4% | 2 |
| M4L4 (gate-off) | 21.5K | 6.1% | 2 |
| M1L8 (default) | 18.4K | 12.4% | 3 |
| M4L4 (gate-on) | 22.5K | 8.6–8.7% | 2 |
| M4L8 (gate-off) | 43.0K | 15.4% | 2 |
| M4L8 (gate-on) | 45.1K | 25.7% | 2 |
| M1L16 (gate-off) | 36.9K | **31.7%** | 2 |
| Depthwise separable conv k=8 | 131.1K | 78.3% | 2 |
| rank-1 LoRA | 32.8K | **87.4%** | 2 |

Read this table as a sequence of single moves: each row, relative to the ones above, trains one more component or deepens the graft. The placebo control offers an early reassurance — a randomly initialized branch closes 0.0%, so everything above it is learned.

The first finding is that the bulk of the gain comes from an unexpectedly humble component: the per-layer output-gain scalar. Training only these 8 scalars on the M1L8 skeleton already closes 10.0%; switching to a multi-scale skeleton while still training only the 8 scalars gives an essentially identical 10.1%. At sixteen layers, 16 scalars alone close 26.4%. In other words, at both depths the pure gain channel accounts for 81% to 83% of the full structure's recovery, with the complete spectral apparatus — feature maps, kernel scales, gating — adding only 2.4 to 5.4 points on top. More intriguing is how the gain channel itself scales: doubling depth lifts closure from 10.0% to 26.4%, far more than linear. Section 5.3 shows what this channel really is: a calibration of the output.

The second finding is the dominance of depth. The single-scale family rises monotonically with grafting depth and steepens as it goes: 5.4% at four layers, 12.4% at eight, 31.7% at sixteen. Stated in budget-matched form the conclusion turns sharper still — redirecting the M4L8 budget (45.1K parameters, 25.7%) into depth yields M1L16, which uses fewer parameters (36.9K) and closes more (31.7%); redirecting the M4L4 gate-on budget (22.5K, 8.6%) into depth yields M1L8, again fewer parameters (18.4K) and more closure (12.4%). An honest boundary needs stating: at four layers the raw cross-cell comparison runs the other way (8.6% for the gated multi-scale cell against 5.4% for single-scale), which is exactly why we state the claim in budget-matched rather than point-wise form.

The third finding concerns multi-scale structure and gating: genuine, but secondary. Without gating, multi-scale adds 0.7 and 3.0 points at depths four and eight; the gate's contribution grows with depth, from 2.5 points at four layers to 10.3 at eight. The trained gate does learn selectivity — its average entropy is only 62.8% of uniform, and perturbing the query token moves the gate 113 times as strongly as perturbing far-context tokens. Kernel scales in the multi-scale configurations drift modestly from initialization (−10% to +14%), suggesting the initial spacing was largely sound and training plays a refining role; in the single-scale configurations the trajectories beyond the gain are smaller still (see the supplementary material).

The fourth finding carries the most weight. A budget-matched rank-1 LoRA — a generic adapter with no spectral structure, simply a low-rank pair alongside the attention output projection — closes 87.4% with 32.8K parameters, 2.8 times the spectral family's best; even a plain depthwise convolution, given seven times the budget, reaches 78.3%. Flanked by two generic adapters, the conclusion is hard to escape: the gap between local and full attention is mostly a generic capacity deficit, not a far-field information gap that only spectral structure can express — the spectral form is not a necessary inductive bias for this task.

One niche, however, deserves naming. Below roughly four thousand parameters, generic adapters cannot even be instantiated — a rank-1 LoRA on a single layer's output projection already costs 4,096 — while 16 gain scalars close 26.4%. In the ultra-low-budget regime, spectral gain calibration has no generic competitor, and future matched-budget adaptation studies should include it as a control.

### 5.3 What the Gain Is: Smoothing, Not Retrieval

Perplexity can only say that overall language modeling improved; it cannot say whether the model recovered specific distant facts. The needle-in-a-haystack test, placing the answer at controlled distances, paints a subtler picture.

**Table 3: NIAH answer-token NLL (w=128, 2,048 tokens; 33 needles, 11 per bucket; survival asserted per example, aggregate rate 1.0; ± is the standard deviation across examples; parentheses give closure of the local–full NLL gap — negative means the correction is worse than local)**

| Configuration | Near (≤128) | Mid (129–256) | Far (>256) |
|---|---|---|---|
| Full | 0.82±0.15 | 0.80±0.17 | 0.91±0.11 |
| Local | 2.37±0.83 | 9.35±1.79 | 9.75±0.25 |
| + M1L8 | 2.60±0.70 (−15.2%) | 8.62±2.49 (+8.6%) | 9.72±0.17 (+0.4%) |
| + M4L4 | 2.57±0.70 (−12.8%) | 8.71±2.58 (+7.5%) | 9.75±0.18 (+0.0%) |
| + M4L8 | 2.49±0.66 (−7.6%) | 8.64±2.58 (+8.4%) | 9.79±0.16 (−0.4%) |
| + M1L16 | 2.43±0.54 (−4.3%) | 8.73±2.40 (+7.3%) | 9.85±0.16 (−1.1%) |
| + Gain-only (M1L8) | 2.59±0.70 (−14.3%) | 8.61±2.50 (+8.7%) | 9.69±0.17 (+0.7%) |
| + Gain-only (M1L16) | 2.42±0.53 (−3.5%) | 8.66±2.35 (+8.2%) | 9.68±0.16 (+0.8%) |

Look inside the window first. The evidence is right there and local attention was already doing well (2.37 against full attention's 0.82), yet all six corrected configurations make it slightly worse, by 3.5% to 15.2%. The cause is the injection itself: the far branch overlays a smooth average of out-of-window history onto positions where local attention was already sharply focused, diluting that focus. The cost narrows with depth — from −15.2% at M1L8 to −4.3% at M1L16 — which we report as a descriptive fact without a verified mechanism. Especially telling is that the gain-only probes pay almost exactly the same cost as the full structures (−14.3% versus −15.2% at eight layers; −3.5% versus −4.3% at sixteen): the cost travels with the gain channel, not with the kernel.

Just beyond the window, the picture reverses. Local attention is blind at this distance while the exponential kernel's weight has not yet decayed away, and here all six configurations improve answer NLL by 7% to 9% of the local–full gap. This is the only region where the branch genuinely helps retrieval-style prediction, and again the mechanism runs through the gain channel — the gain-only probes carry the same signature.

Farther out there is nothing. Beyond two windows, no configuration improves by more than 0.8%, and the deepest configuration lands 1.1% worse. One apparent exception deserves a direct look: at 1,024 tokens, M1L16 closes +15.0% of the far-bucket gap (NLL from 10.66 down to 9.21). Yet both numbers remain an order of magnitude above the full-attention baseline, and the effect vanishes entirely at 2,048 tokens (−1.1%) — we read it as improved prior calibration over distant generic text, not evidence retrieval. This mirrors the attribution of Section 5.2: what the spectral correction recovers is a smooth average of out-of-window information, enough to gently adjust the model's expectations but not to pick out one specific distant fact. That would require sharp, content-selective attention — exactly what fixed exponential kernels with positive feature maps cannot express by design.

Task metrics supply the other half of the same conclusion. Under official LongBench metrics, all three compared models sit near the floor: full attention averages 0.080, local 0.033, corrected 0.033. The window's real cost lands precisely on tasks that demand long-range integration — gov_report's ROUGE-L falls from 0.137 to 0.003, triviaqa's F1 from 0.197 to 0.0 — and the failure mode is degenerate repetition (for instance "Answer: Answer: …") rather than graceful degradation. On the remaining five tasks all three models generate verbatim-identical floor-level text, where closure is undefined; on the three tasks where local and full differ, the correction closes at most 1.3%. A needle-aware RULER retest — whose full-attention arm passes the positive control at perplexity 1.56, confirming the protocol can detect retrieval when it exists — likewise yields only 3.0% closure. Perplexity gains do not convert into task capability.

### 5.4 Efficiency

**Table 4: Prefill throughput slowdown (each row measured pairwise against a same-session local anchor on the same GPU)**

| Configuration | @512 (tok/s) | @1,024 | @2,048 | @4,096 | Decode @2,048 |
|---|---|---|---|---|---|
| + M1L8 (default) | 2.9× (12,790) | 2.5× (12,020) | 2.0× (9,961) | out of memory | ≈1.0× (115.4 vs 115.2) |
| + M1L16 | 4.8× (7,641) | 4.1× (7,499) | 3.0× (6,640) | out of memory | ≈1.0× (114.8 vs 115.1) |
| + M4L8 | 8.7× (4,541) | 6.7× (4,388) | 5.0× (4,046) | out of memory | ≈1.0× (114.0 vs 114.8) |

The three rows' same-session local anchors at 512 tokens are 36,956, 36,991, and 39,532 tok/s respectively; the slowdown ratios reproduce across sessions within about 6%. Three patterns emerge. The slowdown amortizes with length — local attention's own cost grows with sequence length while the far branch's O(n log n) share shrinks. It scales with the branch's workload (patched layers times scales: 8, 16, 32 units), reaching 2.9×, 4.8×, and 8.7× at 512 tokens. And the bottleneck is memory traffic from the FP32 FFT intermediates: peak memory at 2,048 tokens rises from 3.4GB for local to 13.1GB for M1L8 and 20.2GB for M1L16, and beyond 4,096 tokens every spectral configuration exceeds 24GB (the local arm itself still runs at 12,457 tok/s).

Decoding tells the opposite story. With the far branch structurally off, the corrected model's per-token compute is identical to local by construction; six paired measurements across two sessions fall between 0.99 and 1.10 with median 1.004 — statistically indistinguishable from local. The decode column of Table 4 prints the clean-session pairs.

## 6. Conclusions and Outlook

Returning to the opening question — how much of the out-of-window information can a frozen local-attention model recover with a few tens of thousands of retrofitted parameters? — this paper's complete answer is: a real and stable portion, up to 31.7% of the perplexity gap, but both the composition and the character of that portion deserve a careful look. The gains are dominated by the learned output gains and the grafting depth (16 gain scalars alone close 26.4%, against 31.7% for the full structure); multi-scale kernels and gating contribute genuinely but secondarily. In character, the gain is statistical smoothing of token likelihoods — it improves prediction just beyond the window, slightly sacrifices in-window precision, cannot perform far-distance retrieval, and does not transfer to downstream task metrics. Under budget-matched comparison, generic low-rank adaptation's 87.4% against 31.7% makes the further point that this gap is mostly generic capacity, for which spectral structure is not a necessary bias.

For practitioners the advice is correspondingly direct. If the goal is to lower a frozen model's perplexity, choose LoRA — simpler and stronger. If committed to the spectral form, spend the budget on grafting depth rather than kernel diversity. And no such correction should be expected to restore far-distance retrieval. The single exception is the ultra-low-budget niche: below roughly four thousand parameters, generic adapters cannot be instantiated at all, and gain scalars have no competitor there.

For longer-term research, the negative results point forward as well. The far-retrieval failure is structural: fixed kernels with positive feature maps cannot express content selectivity. Retrofitted global pathways that want retrieval capability will likely need addressable memory — compressed KV stores or explicit retrieval — rather than smooth spectral aggregation. At the same time, the one region where the correction does improve NLL lies just past the window's edge, exactly where local attention goes blind, suggesting that calibrating representations at the window's edge is an underappreciated source of gains in low-budget adaptation.

The limitations deserve honest statement. Experiments span 1B–1.5B models and training lengths of at most 2,048 tokens, so extrapolation to larger models or longer contexts calls for caution. The retrieval- and task-level analyses (NIAH, LongBench, RULER) were conducted on LLaMA 3.2-1B alone, with GPT2-XL serving only to verify that perplexity gains generalize across architectures. The evaluation uses a single window size (w=128); a joint sweep of window and kernel scales is left to future work. LongBench coverage is 7 of 15 English tasks, with the three-way protocol fully matched and no reason to expect the conclusion to change with a larger task set. The multi-scale configuration at sixteen layers was not tested, but the dominance conclusion — LoRA's 87.4% against the family's best 31.7% — does not depend on it. Finally, the spectral branch's prefill overhead has seen no kernel-level optimization; the measured slowdowns are upper bounds of an unoptimized implementation, not lower bounds of the method. All implementation-level debugging records, the nine-reversal ledger, and the machine-executed assertion list are collected in the supplementary material.

---
