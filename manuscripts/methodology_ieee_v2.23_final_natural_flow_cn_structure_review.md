# methodology_ieee_v2.23 中文结构说明

源文件：`manuscripts/methodology_ieee_v2.23_final_natural_flow.md`

说明：这份文件不是逐句翻译，而是按当前英文手稿的段落顺序，对每一段的核心内容和结构作用做中文总结，方便检查论文整体 flow 是否合理。

---

## III. SYSTEM OVERVIEW AND PROBLEM FORMULATION

### 段落 III-0：系统与问题总引入

**对应内容：** Section III 开头第一段。

**内容总结：** 介绍本文研究的是一种磁驱动、非接触式康复系统。该系统希望在避免机器人和患者刚性接触的同时，保留主动运动训练任务。控制问题被定义为一个强化学习任务，机器人通过控制磁性微型机器人来调节患者交互难度。

**结构作用：** 从系统层面引出全文研究对象，并把硬件系统和 RL 控制问题连接起来。

---

### III.A Hardware Platform

#### 段落 III-A-1：硬件三层结构

**对应内容：** 硬件平台第一段。

**内容总结：** 说明平台由三层组成：底层 UR10 机械臂和末端磁铁负责驱动；中间 acrylic 板和无源磁性微型机器人构成人机交互层；上方 RGB 相机负责感知手和微型机器人的位置，并由主机向 UR10 发送控制命令。

**结构作用：** 交代真实系统的物理组成，让读者知道仿真和控制策略最终对应什么硬件平台。

#### 段落 III-A-2：安全性与可复现性

**对应内容：** 硬件平台第二段。

**内容总结：** 解释这个物理设计为什么适合康复实验。Acrylic 隔离界面可以降低直接接触设备带来的冲击、牵拉和局部压迫风险；系统只依赖协作机器人、相机和磁铁，不需要穿戴设备、力传感器、电池或定制执行器，因此更容易搭建和复现。

**结构作用：** 说明硬件设计的合理性，不只是描述设备，而是强调安全和实验可复现两个论文价值点。

---

### III.B Problem Formulation

#### 段落 III-B-1：康复任务定义为 pursuit-evasion

**对应内容：** Problem Formulation 第一段。

**内容总结：** 将康复任务表述为平面工作空间中的追逃游戏。患者用手追踪由机器人控制的磁性微型机器人，目标是接近到接触半径内。与静态 reaching 不同，微型机器人会主动逃逸，因此任务要求持续视觉-运动协调，而不是单次点到点运动。

**结构作用：** 建立本文核心任务形式，为后面 ZPD、RL reward 和 league training 提供任务背景。

#### 段落 III-B-2：ZPD 目标距离区间

**对应内容：** Problem Formulation 第二段。

**内容总结：** 定义 Zone of Proximal Development，即手和微型机器人之间的目标距离范围。机器人不能逃得太远，也不能过度让步，而是要让交互保持在一个既有挑战又可完成的范围内。患者速度、策略、延迟和噪声会变化，因此固定难度设置不够。

**结构作用：** 把康复目标从“追到目标”转化为“保持合适距离”，这是后面 reward、TIS 指标和实验评价的中心。

#### 段落 III-B-3：传统控制器局限

**对应内容：** Problem Formulation 第三段。

**内容总结：** 说明传统控制方法不适合这个问题。人工势场或规则式阻抗控制通常依赖当前状态的手工反馈规则，难以预测 interception 这类策略性追踪行为，也难以适应不同患者的运动模式。在 tremor、延迟反应或 bradykinesia 下，固定规则会进一步失效。

**结构作用：** 建立为什么不能只用经典控制器，从而为 RL 方法提供必要性。

#### 段落 III-B-4：RL 形式化与剩余挑战

**对应内容：** Problem Formulation 第四段。

**内容总结：** 说明该问题适合用 RL 表述，因为控制器需要在患者行为变化下优化长期 ZPD 目标。机器人学习一个 evasion policy，在距离保持在治疗范围内时获得奖励，在逃太远或太容易被抓住时受到惩罚。段末指出剩余挑战是如何在不假设固定或完全可观测患者响应模型的情况下学习这个策略。

**结构作用：** 从问题定义自然过渡到方法部分。这里不提前展开具体方法，只提出方法需要解决的核心挑战。

---

## IV. METHODOLOGY

### 段落 IV-0：方法总览

**对应内容：** Methodology 开头段。

**内容总结：** 方法部分回应上一节提出的挑战。患者行为在策略和运动执行层面都会变化，因此本文把问题拆成三个方面：仿真患者分布、观测表示、训练分布。对应方法分别是 CMD-DR、dual-stream encoder with auxiliary head，以及 iterative league training。

**结构作用：** 提供方法章节的整体框架，并把三个方法模块和前文挑战对应起来。

---

### IV.A Cognitive-Motor Decoupled Domain Randomization

#### 段落 IV-A-1：CMD-DR 的两层随机化

**对应内容：** IV.A 第一段。

**内容总结：** 说明 robot policy 必须同时面对患者策略差异和运动执行差异。CMD-DR 将两类变化分开处理：先从 opponent pool 采样 hand policy，决定每一步想要的手部位移；再通过运动和感知约束把这个 intended displacement 转换成实际执行位移。随后引出低通滤波公式，用来模拟肌肉惯性。

**结构作用：** 介绍 CMD-DR 的核心思想，即将“想怎么动”和“身体实际怎么动”分离。

#### 段落 IV-A-2：低通滤波公式

**对应内容：** Equation (1) 及其说明。

**内容总结：** 用一阶低通滤波模拟肌肉惯性。原始期望位移不会直接执行，而是与上一时刻状态混合，产生平滑后的动作。

**结构作用：** 给 CMD-DR 的 motor execution 层提供第一个具体数学机制。

#### 段落 IV-A-3：加速度裁剪公式

**对应内容：** Equation (2) 及其说明。

**内容总结：** 通过裁剪相邻时间步之间的位移变化，限制手部运动的突变，避免不真实的突然加速。

**结构作用：** 给 CMD-DR 的 biomechanical constraint 增加运动平滑和物理合理性。

#### 段落 IV-A-4：噪声、延迟和参数表

**对应内容：** IV.A 公式后说明和 Table I。

**内容总结：** 进一步加入视觉位置估计噪声和神经/运动延迟。Table I 列出各约束的参数范围，包括肌肉惯性、最大加速度、观测噪声和延迟帧数。段末强调，分离意图和执行可以让 robot policy 接触更丰富的 patient-like behaviors，减少对单一仿真手模型的依赖。

**结构作用：** 完成 CMD-DR 的具体组成，并为下一节“这些变化需要从观测中推断”埋下逻辑基础。

---

### IV.B Dual-Stream Encoder with Future Dynamics Head

#### 段落 IV-B-1：从 CMD-DR 过渡到观测推断

**对应内容：** IV.B 第一段。

**内容总结：** CMD-DR 增加了训练中患者行为的多样性，但这些变化仍然必须由 policy 从观测中推断。当前手的位置不能区分主动追踪、感知噪声、运动延迟、tremor-like perturbation 或 slow response。因此观测被编码成两个 stream，一个处理当前几何状态，一个处理近期手部运动历史。

**结构作用：** 自然承接 IV.A，说明为什么生成多样行为之后还需要 temporal encoder。

#### 段落 IV-B-2：观测向量拆分

**对应内容：** Equation (3) 前后。

**内容总结：** 每个时间步的 44 维观测被拆成两个部分：12 维 scalar vector 和 32 维 temporal buffer。

**结构作用：** 给 dual-stream encoder 的输入结构做形式化定义。

#### 段落 IV-B-3：scalar component 定义

**对应内容：** Equation (4) 及变量说明。

**内容总结：** scalar component 表示当前交互的瞬时几何状态，包括微型机器人和手的位置、两者距离、到边界的距离、当前 stride 和上一动作。

**结构作用：** 解释空间状态分支输入什么，以及这些变量如何描述当前几何关系。

#### 段落 IV-B-4：temporal buffer 的作用

**对应内容：** temporal buffer 段。

**内容总结：** temporal buffer 包含最近 16 帧相对手部位移，总共 32 维。这个历史窗口可以帮助模型估计运动方向、反应延迟和振荡行为。使用相对位移而不是绝对位置，使表示对平移更鲁棒，并聚焦于手如何运动。

**结构作用：** 说明为什么历史信息对 patient intent inference 有用。

#### 段落 IV-B-5：MLP + GRU 双流融合

**对应内容：** dual-stream encoding 段。

**内容总结：** scalar vector 由 MLP 编码，temporal buffer 由 GRU 编码。两个 stream 的输出拼接后进入 fusion MLP，生成 policy head 和 value head 共用的表示。

**结构作用：** 交代网络结构如何把空间状态和时间历史结合起来。

#### 段落 IV-B-6：为什么需要 auxiliary head

**对应内容：** auxiliary head 引入段。

**内容总结：** 即使 motion history 被输入给 policy，也不保证 recurrent representation 一定学到 patient dynamics。ZPD reward 只提供控制层面的信号，并不直接监督 encoder 如何表示患者运动。因此引入 future-dynamics head，提供更密集的学习信号。

**结构作用：** 从 representation availability 过渡到 representation learning，解释 auxiliary task 的必要性。

#### 段落 IV-B-7：future prediction 目标

**对应内容：** Equation (5) 及说明。

**内容总结：** auxiliary head 预测未来 8 步患者相对位移，并估计 near-catch risk。由于未来位移可以直接从 rollout 中获得，因此不需要额外标注。轨迹预测损失用预测位移序列和真实位移序列之间的误差定义。

**结构作用：** 定义 auxiliary task 的具体监督目标。

#### 段落 IV-B-8：总训练目标与共享梯度

**对应内容：** Equation (6) 及说明。

**内容总结：** 总损失由 PPO loss、trajectory loss 和 risk-prediction loss 组成。由于 future-dynamics head 与 policy 共享 encoder，auxiliary loss 的梯度会更新空间和时间分支，促使 encoder 学到患者惯性、延迟和交互动态。

**结构作用：** 说明 auxiliary head 不只是额外输出，而是通过共享 encoder 改善 policy 表示。

---

### IV.C League Training

#### 段落 IV-C-1：为什么 motion representation 还不够

**对应内容：** IV.C 第一段。

**内容总结：** 有用的运动表示还不足以保证策略鲁棒。如果训练只暴露给机器人狭窄或不稳定的 hand behaviors，策略仍然会过拟合或不稳定。同时训练 robot 和 hand policies 会造成非平稳学习问题，因为任一方更新都会改变另一方面对的环境。因此本文采用交替冻结训练，一方训练时另一方固定。

**结构作用：** 从 representation 问题自然进入 training distribution 问题，说明 league training 的必要性。

#### 段落 IV-C-2：康复任务不是纯竞争游戏

**对应内容：** IV.C 第二段。

**内容总结：** 康复场景中机器人不是要击败患者，而是要维持合适挑战水平。手部策略被称为 hand policy，可以是 scripted 或 learned controller，并在 robot training 中作为 opponent。因为单一 hand policy 无法覆盖临床中的运动差异，所以维护一个包含不同训练阶段 hand policies 的 opponent pool。

**结构作用：** 将 league training 从普通 self-play 区分出来，强调目标是覆盖患者能力差异，而不是单纯赢。

#### 段落 IV-C-3：迭代训练流程

**对应内容：** Algorithm 1 前后。

**内容总结：** 每次 league iteration 包含两个阶段。robot phase 中固定 opponent pool，机器人用 PPO 对池中采样的 hand policies 训练。hand phase 中固定当前 robot policy，从上一代 hand policy warm start 并继续训练。训练后的新 hand policy 加入 pool，供下一轮 robot training 使用。

**结构作用：** 给出 iterative league training 的整体流程。

#### 段落 IV-C-4：PFSP 采样规则

**对应内容：** PFSP 段和 Equation (7)。

**内容总结：** uniform sampling 低效，因为太弱的 hand policy 学习信号少，太强的 hand policy 会过早结束 episode。最有信息量的是对当前 robot 有挑战但不是完全不可行的 opponents。本文使用基于 normalized episode length 的 PFSP 采样规则，将中等难度 opponents 赋予更高概率，并通过 probability floor 保留旧 opponents。

**结构作用：** 解释 league 训练中 opponent sampling 如何集中训练压力，同时避免遗忘旧策略。

#### 段落 IV-C-5：保留 scripted hand 的原因

**对应内容：** IV.C 最后一段。

**内容总结：** learned hand policies 可能带有仿真中的人工模式，机器人如果利用这些模式，可能降低真实康复相关性。为降低这种风险，heuristic scripted hand 始终保留在 pool 中，使 robot 持续接触简单、带噪声、接近真实患者追踪行为的 hand behavior。

**结构作用：** 补充 league training 的安全性和 sim-to-real 合理性考虑。

---

## V. EXPERIMENTS AND RESULTS

### 段落 V-0：实验总目标

**对应内容：** Section V 开头。

**内容总结：** 实验部分评估三个方面：对 learned hand behaviors 的鲁棒性、temporal patient-history encoding 的贡献，以及在真实 UR10 平台上的实时部署。

**结构作用：** 简洁说明实验章节覆盖范围，不展开重复 roadmap。

---

### V.A Experimental Setup

#### 段落 V-A-1：仿真环境与 ZPD 设置

**对应内容：** V.A 第一段。

**内容总结：** 所有仿真实验都使用 Section III 中定义的 pursuit-evasion rehabilitation environment。机器人控制磁性微型机器人，手由 scripted pursuit controller 或 learned hand policy 控制。治疗目标为 ZPD 距离区间 3.5 到 5.5 workspace units，距离落在该区间内的 step 计为 in-zone。

**结构作用：** 统一实验环境和任务设置，保证后续结果可比较。

#### 段落 V-A-2：评价指标

**对应内容：** V.A 第二段。

**内容总结：** 主要指标是 Time-In-Zone，即 episode 中处于 ZPD band 内的时间比例。还报告 ZPD coverage、episode length、catch rate 和 across learned hand pool 的 empirical robustness。TIS 衡量持续治疗交互，ZPD coverage 衡量已发生 episode 部分的质量，episode length 衡量 survival，catch rate 识别过于简单、被过快抓住的交互。

**结构作用：** 定义结果部分使用的指标，并说明每个指标对应的康复意义。

#### 段落 V-A-3：实验协议与 ablation 设置

**对应内容：** V.A 第三段。

**内容总结：** protocol comparison 用于隔离 opponent diversity 的影响，包括 scripted-only、single-hand 和 league 三种 robot training protocols。representation ablation 比较 MLP、GRU 和 GRU+Aux 三种策略表示，用于评估 temporal history 和 auxiliary dynamics head 的作用。

**结构作用：** 为 V.B 的 league robustness 结果和 V.C 的 network ablation 结果做实验设计铺垫。

---

### V.B League Training and Robustness Evaluation

#### 段落 V-B-1：league evaluation 的目的和设置

**对应内容：** V.B 第一段。

**内容总结：** league evaluation 检验 opponent-pool training 是否能减少单一 hand model 训练带来的脆弱特化。机器人训练 10 代，hand pool 随 learned hand policies 扩展。机器人不接收 opponent identity，只能从空间状态和近期位移历史中推断当前 hand behavior。

**结构作用：** 明确 league 结果要回答的问题，即无 opponent ID 情况下的鲁棒性。

#### 段落 V-B-2：平均表现和困难样本表现都改善

**对应内容：** V.B 第二段。

**内容总结：** league training 同时改善平均行为和 difficult-case behavior。cross-iteration validation matrix 将每一代 robot 与每个 learned hand policy 配对评估。早期 learned hand policies 相对容易调节，后期 policies 带来更困难的追踪策略。后期 robot generations 在矩阵中更大范围表现改善，说明 league 不只是适应最终 hand policy。

**结构作用：** 用 Fig. 1 的矩阵结果说明 league training 改善的是跨 hand policy 的整体鲁棒性。

#### 段落 V-B-3：lower-tail robustness 的意义

**对应内容：** V.B 第三段。

**内容总结：** 对康复来说，lower-tail robustness 尤其重要。平均 TIS 提高的同时，worst-hand TIS 和 CVaR20 也提高，说明改进不仅发生在平均性能上。frontier view 显示后期 robot generations 更能在困难 hand policies 上保持 ZPD regulation。

**结构作用：** 把结果解释从平均值推进到困难病例鲁棒性，呼应康复场景中的个体差异。

#### 段落 V-B-4：最终代 failure mode 分析

**对应内容：** V.B 第四段。

**内容总结：** final-generation results 分解了剩余失败模式。一些 hand policies 导致 too-close episodes，使任务太容易；另一些导致 too-far interactions，使任务过难或不可达。league policy 在这些不同失败模式下仍保持有用的 ZPD coverage，说明策略不是只针对某一种 opponent 调整。

**结构作用：** 从 robustness 指标进一步解释失败机制，说明 league 策略的平衡性。

#### 段落 V-B-5：PFSP sampling traces 的训练机制解释

**对应内容：** V.B 第五段。

**内容总结：** PFSP sampling traces 将最终 balanced performance 与 opponent sampling 形成的 curriculum 联系起来。训练过程中，sampling probability 会转向当前 robot generation 仍然觉得有信息量的 hand policies，同时 probability floor 保留早期 policies，避免遗忘。

**结构作用：** 解释为什么 league training 能产生前面观察到的鲁棒性。

#### 段落 V-B-6：Fig. 1 图注

**对应内容：** Fig. 1 caption。

**内容总结：** Fig. 1 包含五个子图：cross-iteration TIS matrix、mean/worst/CVaR20 curves、robustness frontier、final-generation failure decomposition，以及 PFSP sampling probability snapshots。

**结构作用：** 图注自包含地说明 Fig. 1 各面板内容，正文则负责解释这些结果的意义。

#### 段落 V-B-7：自动 hand-controller policy comparison 设置

**对应内容：** Table II 前说明段。

**内容总结：** 对 scripted-only、single-hand 和 league robot policies，在 scripted hand 与 learned hand policy 两种 automated controller 下进行比较。mouse-controlled human-in-the-loop 测试单独保留，因为它包含人类反应时间和主动策略，不只是另一种自动 hand controller。

**结构作用：** 引出 Table II，并说明为什么 mouse test 不并入该表。

#### 段落 V-B-8：Table II 内容

**对应内容：** Table II。

**内容总结：** 表格比较三种 robot policy 在 scripted-hand 和 learned-hand 条件下的 TIS、ZPD coverage、episode length 和 catch rate。league policy 在两个测试条件下都比较均衡，而 scripted-only 和 single-hand baseline 各自在另一类 hand controller 上出现明显退化。

**结构作用：** 提供 automated hand-controller generalization 的定量证据。

#### 段落 V-B-9：Table II 结果解释

**对应内容：** Table II 后解释段。

**内容总结：** 窄 hand model 训练会导致互补失败模式。scripted-only robot 在 scripted controller 上还可以，但面对 learned hand policy 时经常被抓住；single-hand robot 则相反。league-trained robot 不一定每个单项指标都最好，但避免了跨条件崩溃。对于 adaptive rehabilitation，均衡表现比只优化单一测试条件更重要。

**结构作用：** 将表格数字转化为论文论点，即 league training 提供更稳健的跨 hand-controller 表现。

---

### V.C Network Ablation and Auxiliary Dynamics Analysis

#### 段落 V-C-1：ablation 的核心问题

**对应内容：** V.C 第一段。

**内容总结：** opponent diversity 不能单独决定 robot 是否能使用观测信息。ablation 问题是 robot 仅凭当前几何状态是否足够调节难度，还是需要近期 patient-motion history。比较对象包括 MLP、GRU 和 GRU+Aux。

**结构作用：** 从 league robustness 转到 representation design，说明 V.C 的研究问题。

#### 段落 V-C-2：temporal history 的主要性能贡献

**对应内容：** V.C 第二段。

**内容总结：** Fig. 2 显示主要性能提升来自引入 temporal interaction history。GRU 和 GRU+Aux 都比 MLP 学到更长、更高 reward 的交互。GRU-only 和 GRU+Aux 的整体学习趋势接近，说明在这次运行中 temporal sequence modeling 是主要贡献因素。

**结构作用：** 给出 ablation 的主要结论，即 GRU temporal encoding 比单帧 MLP 更关键。

#### 段落 V-C-3：auxiliary prediction 的诊断意义

**对应内容：** V.C 第三段。

**内容总结：** auxiliary prediction examples 用来诊断 learned temporal representation。future-dynamics head 能捕捉短期运动方向，但随着预测 horizon 增加误差逐渐扩大。这说明 auxiliary task 的价值更多在于鼓励并暴露 patient-motion structure，而不是在本次实验中带来明显额外 reward 提升。

**结构作用：** 解释为什么即使 GRU+Aux 没显著超过 GRU，auxiliary head 仍有表示学习和可解释价值。

#### 段落 V-C-4：Fig. 2 图注

**对应内容：** Fig. 2 caption。

**内容总结：** Fig. 2 展示 MLP、GRU、GRU+Aux 的 smoothed reward 和 episode length 曲线，以及 auxiliary future-prediction examples 和 horizon-dependent trajectory error。

**结构作用：** 图注说明 ablation 图的组成，正文负责解释 temporal history 和 auxiliary task 的意义。

---

### V.D Real-World Deployment Pipeline

#### 段落 V-D-1：部署实验范围

**对应内容：** V.D 第一段。

**内容总结：** deployment experiment 将仿真训练得到的 policy 连接到真实 UR10 平台。由于真实手部运动、相机延迟、磁驱动和安全约束没有完全体现在虚拟训练环境中，本节重点不是成对 sim-to-real benchmark，而是展示 learned policy 能否运行在实时感知-控制栈中。

**结构作用：** 清楚限定 real-world 部分的证据范围，避免把它写成完整 sim-to-real 性能评估。

#### 段落 V-D-2：真实系统观测与异步线程需求

**对应内容：** V.D 第二段。

**内容总结：** 真实系统使用与仿真相同的 44 维观测，包括 RTDE 机器人位置、相机检测的手和微型机器人位置、边界特征、上一动作和 displacement-history buffer。感知和控制分线程运行，以避免相机推理延迟影响 20 Hz robot control loop。

**结构作用：** 说明仿真 observation 如何映射到真实系统，并解释异步架构的必要性。

#### 段落 V-D-3：实时控制线程、队列和 dead reckoning

**对应内容：** V.D 第三段。

**内容总结：** 系统用单元素队列连接 vision thread 和 control thread。vision thread 采集图像、去畸变、检测并推送结果；如果队列满，就替换旧结果。control thread 每个周期非阻塞读取新视觉结果；若没有新帧，则用低通滤波速度估计进行 dead reckoning。最终 observation 输入 PPO policy，二维动作经 inverse homography 映射后通过 servoL 发给 UR10。

**结构作用：** 交代真实部署中如何在低帧率视觉和固定频率控制之间保持稳定闭环。

---

## 全文结构主线总结

1. **Section III** 先说明硬件系统和康复任务，再把问题抽象为 ZPD-guided pursuit-evasion RL。
2. **Section IV.A** 解决仿真患者分布问题，即如何生成多样但合理的 patient-like behaviors。
3. **Section IV.B** 解决观测表示问题，即 policy 如何从当前几何状态和历史运动中推断患者动态。
4. **Section IV.C** 解决训练分布问题，即如何避免单一 hand policy 或同步训练导致的过拟合和非平稳性。
5. **Section V.A** 统一实验设置、评价指标和对比协议。
6. **Section V.B** 证明 league training 提高跨 learned hand policies 和 hand-controller types 的鲁棒性。
7. **Section V.C** 证明 temporal history 是策略表示中的关键因素，并解释 auxiliary head 的诊断价值。
8. **Section V.D** 说明仿真策略如何接入真实 UR10 感知-控制系统。

## 结构审阅时可重点检查的问题

- III.B 到 IV 是否自然形成“问题 → 方法”的关系。
- IV.A、IV.B、IV.C 是否分别对应 simulation variability、observation inference、training distribution 三个层次。
- V.B 和 V.C 是否分别回答 opponent diversity 与 representation design 两个实验问题。
- V.D 是否只是部署 pipeline 说明，而不是被误读为完整 sim-to-real benchmark。
- Fig. 1 和 Fig. 2 的正文是否负责解释结果意义，而不是重复图注。
