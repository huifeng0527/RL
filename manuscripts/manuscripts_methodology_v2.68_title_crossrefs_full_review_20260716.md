# Editorial Decision Package — v2.68 Full Review

**Manuscript:** *Adaptive Difficulty Regulation in Magnetic Microrobot-Based Upper-Limb Rehabilitation via League-Trained Reinforcement Learning*  
**Reviewed file:** `manuscripts_methodology_v2.68_title_crossrefs.docx`  
**Review date:** 2026-07-16  
**Mode:** academic-paper-reviewer `full`  
**Panel:** EIC + Methodology + Rehabilitation Domain + Practical Systems + Devil’s Advocate

## 1. Editorial decision

**Formal contract outcome:** `reject_or_major_revision`  
**Internal pre-submission decision:** **Major Revision — not ready for submission**

论文的研究问题、非接触磁驱平台和 league-training 方向具有发表潜力，正文也对临床边界保持了较好的克制。不过，当前版本存在会阻断投稿的实验来源与方法描述不一致。最严重的问题不是语言，而是论文所描述的 observation、auxiliary objective、league policy 和 physical deployment policy 与存档实验配置没有形成一一对应关系。若不先完成 provenance audit、训练预算校正和 physical rollout 重新归类，真实期刊很可能作出 desk reject 或 reject。

## 2. Contract scoring matrix

| Reviewer | D1 Methodology | D2 Domain | D3 Coherence | D4 Cross-disciplinary | D5 Writing | Recommendation |
|---|---|---|---|---|---|---|
| EIC | warn | pass | warn | pass | pass | Major Revision |
| Methodology | **block** | warn | **block** | warn | warn | reject_or_major_revision |
| Domain | warn | warn | pass | warn | pass | Major Revision |
| Systems | warn | pass | warn | warn | pass | Major Revision |
| Devil’s Advocate | **block** | warn | warn | warn | warn | reject_or_major_revision |

- **F1 triggered:** mandatory dimension has `block`.
- **F2 triggered:** all five reviewers have at least two mandatory dimensions at `warn` or worse.
- **F3 not triggered:** D4 has no `block`.
- **F0 not triggered:** mandatory dimensions do not all pass.

## 3. Strengths agreed by the panel

1. **研究问题明确。** 论文把 adaptive difficulty formulation 转化为 hand–target distance regulation，适合建立可测量的控制问题。
2. **平台构型有辨识度。** 轻量、无动力的桌面目标与人体没有刚性耦合，位于纯虚拟训练和接触式 rehabilitation robot 之间。
3. **league robustness 的研究方向合理。** 相比单一 virtual hand，历史 hand pool 能减少明显的 opponent-specific specialization。
4. **temporal representation 的动机充分。** 当前几何状态不能表达加速、转向和 delay；引入 interaction history 符合 pursuit-control 需求。
5. **Discussion 边界控制较好。** 论文明确说明当前证据不支持 clinical efficacy、motor recovery 或 patient benefit。
6. **整体写作结构清楚。** Introduction、System、Methodology、Experiments、Discussion 和 Conclusion 的主线易于理解。

## 4. Priority 1 — Must Fix

### P1.1 建立 claim-to-artifact provenance map

**Sources:** Methodology, EIC, Systems, Devil’s Advocate

为每一项贡献、图、表和 physical rollout 建立内部表格，至少记录：

- run directory
- exact checkpoint
- code commit/source snapshot
- observation dimension and channel semantics
- history mode
- auxiliary mode and loss weights
- robot/hand training steps
- training and evaluation seeds
- test-suite composition
- physical rollout policy checkpoint

**Acceptance criterion:** 每个 figure/table/result 都能对应唯一 config 和 checkpoint；论文中不再把不同 experiment family 写成同一 policy/framework 的统一结果。

### P1.2 解决 44-D 与 140-D observation 的核心矛盾

论文在 pp. 5–6 报告：

- 12-D scalar state
- 16 × 2 recent relative hand displacement
- total observation dimension = 44

但存档结果显示：

- `logs/league_zpd35_55_noid_warm_entropy_10iter_r5m_h1m_gru_noaux/run_config.json` 使用 `history_mode: interaction`
- `src/observation_schema.py:8-23` 定义 interaction history 为 8 channels
- `src/custom_env.py:910-920` 的 channels 包含 relative x/y、distance、distance change、robot move x/y、hand move x/y
- total dimension = 12 + 16 × 8 = 140
- Fig. 4 的来源配置 `logs/ablation_gru_h1_h10_0626_2136/config.json` 同样使用 `history_mode: interaction`

**Required action:** 二选一。

1. 按论文所述 44-D、16×2 observation 重新训练并重做核心实验；或
2. 将 Methodology、equations、Fig. 2、captions 和 deployment description 全部改成实际使用的 140-D、16×8 interaction history。

### P1.3 解决 auxiliary-objective mismatch

当前存档显示：

- final 10-iteration league run 使用 `aux_mode: none`
- physical deployment family 使用 `aux_mode: multi_risk`
- 论文则把 trajectory prediction auxiliary objective 写成 proposed framework 的统一组成

因此，当前结果不能统一归因于 trajectory auxiliary prediction。

**Required action:**

- 如果保留现有结果，明确分开写：no-aux league、auxiliary ablation、multi-risk physical deployment；
- 如果 trajectory auxiliary 是核心贡献，则需用该 objective 重新完成 central league/deployment evaluation；
- 不应把没有使用 auxiliary head 的 league result 解释为 auxiliary objective 的效果。

### P1.4 修复 CMD-DR / Table I 与实现不一致

**Sources:** Methodology, EIC, Devil’s Advocate

稿件 Table I 报告 `alpha = 0.7`、`c_max = 0.15`、`sigma ~ U(0.01, 0.08)` 和 0–3-frame delay，但 repository audit 显示 central archived run 的 active code/config 并不对应这一组合：smoothing 被覆盖、change limit 随 `stride_hand` 变化、noise 为固定值且也作用于 action，默认 healthy path 下 delay 可能未实际生效。

**Required action:**

- 从 exact run config 和对应 code snapshot 重新生成 motor-randomization table；
- 明确哪些随机化在 central league run 中实际启用，哪些只是设计或其他 experiment family 的配置；
- 给出 units、simulation timestep、frame/control rate 和 delay interpretation；
- 若 CMD-DR 仍作为核心贡献，需使用所声明的 execution layer 重新训练并提供 strategy-only、execution-only、coupled perturbation 与 full decoupled randomization 的对照；否则将其降格为未充分验证的 design mechanism。

**Acceptance criterion:** Table I、公式、保存的 config 和 active implementation 一一对应，不再把未启用的 motor filtering/delay 写成 central result 的既定训练条件。

### P1.5 修复 physical deployment provenance

`data/deployment_rollouts/**/metadata.json` 显示：

- physical policies 为 140-D interaction-history policy
- policy family 为 `league_paper_gru_multistep_aux_pfsp_window_20iter`
- 并非 Fig. 3/Table II 所用的 final 10-iteration no-aux run
- 六个 rollouts 使用 iteration 20，三个使用 iteration 10

因此，“nine rollouts of the simulation-trained policy” 暗示单一 checkpoint，但实际 rollouts 混合了两个 checkpoints。

**Required action:**

- 将 iteration 10 和 iteration 20 按 checkpoint、stride 和 target control rate 分组报告；或
- 使用一个明确 checkpoint 和固定 deployment configuration 重新完成一组健康 pilot rollout；
- 报告所有 rollout 的 TIZ/ZPD occupancy、duration、catch、tracking dropout、dead-reckoning fraction、control/camera rate、inference latency、boundary/safety events；
- 明确 Fig. 5 对应哪一个 rollout、checkpoint 和控制设置。

现有 9 个 summary JSON 已能支持初步 aggregate reporting：duration 28.39 ± 7.17 s、camera rate 12.12 ± 0.34 Hz、control-loop rate 17.90 ± 5.63 Hz、mean inference latency 2.237 ± 0.049 ms、p95 3.09 ± 0.03 ms、dead-reckoning fraction 33.23 ± 19.16%、mean distance 4.85 ± 0.46 cm、within-band occupancy 56.01 ± 8.56%、0 safety stops，终止原因为 5 timeout / 4 caught。由于 checkpoint 和控制设置混合，这些值必须分组解释，不能直接当作单一 final policy 的统一性能。

Patient recruitment 不是本轮修订的必要条件；健康 pilot 在 ethics approval/exemption 和 consent 清楚的前提下即可。

### P1.6 校正 league 与 baseline 的训练预算

已核对 archived training status 和 checkpoints：

- R1 robot phase 约为 5M steps；
- R2–R10 每代 robot phase 约为 2M steps；
- final league robot 的累计 robot-training budget 约为 **23M steps**，即 5M + 9 × 2M；
- `scripted_only_5m` 和 `single_hand_h1_5m` baselines 各约为 **5.013M steps**；
- 后续 robot generations 由前代 checkpoint 继续训练。

因此，Table II 的差异仍受到 23M-vs-5M training-budget confound 影响，不能直接归因于 league diversity 或 prioritized sampling。

**Required action:**

- 在正文或 supplementary provenance table 中公开每个 policy 的累计 robot steps；
- 若要作 league mechanism 的因果比较，增加约 23M-step budget-matched baselines，或统一总训练预算；
- 单独比较 uniform pool sampling 与 prioritized sampling，才能支持 PFSP/prioritization 的独立贡献；
- 在预算未匹配前，将当前结果写成受更大累计训练预算影响的 exploratory robustness evidence，而不是 league mechanism 的无混杂因果提升。

### P1.7 增加 simple controller baselines

论文在 Introduction/Problem Formulation 中批评 fixed-speed、distance-reactive 和 potential-field/rule-based controller，但实验只比较 RL training protocols。

至少加入：

- fixed-speed target
- distance-only reactive controller
- distance + relative-velocity controller
- potential-field 或简单 rule controller

在相同 test suite 下报告 TIZ、episode length、too-close/too-far rate、jerk、boundary violations 和 return-to-band time。

### P1.8 补齐 reproducibility 信息

正文必须给出：

- exact ZPD band and catch threshold
- full piecewise reward and weights
- action definition and scaling
- episode horizon and simulation timestep
- hand-policy observation/action/reward
- PPO hyperparameters
- robot and hand steps per phase
- PFSP window, temperature and SPC probability；同时准确描述 two-level sampler：先保留固定 scripted-hand probability，再在 learned-hand pool 内使用 episode-length PFSP 与 uniform exploration mass；`min_prob=0.05` 不应表述为每个 opponent 均有 0.05 的逐项概率下限
- auxiliary optimizer/loss settings
- model-selection rule
- number of independent training seeds and evaluation seeds

### P1.9 修复统计证据

当前 Fig. 4 metrics 来自每个模型 60 evaluation episodes，但未显示独立 training-seed replication。Table II 的十次 seeded trials 主要描述 evaluation variability，不能替代 stochastic RL training variability。

Repository audit 还发现，Table II 声称的 sample SD 与当前 repeated-evaluation artifact 并不一致。例如 automated rows 中，artifact 的 scripted-only vs SPC 约为 0.3146 ± 0.0313、single-H1 vs H1 约为 0.5123 ± 0.0297、league vs H1 约为 0.4497 ± 0.0291，而稿件给出的对应数值和 SD 不同。Manual mouse 的 ± 值也与现有 raw artifact 不完全匹配。必须确认表格究竟来自哪一版 checkpoint/evaluation，以及 ± 表示 sample SD、standard error、confidence interval 还是其他聚合单位。

**Required action:**

- 核心模型至少进行多个 independent training seeds；
- 报告 per-seed results、mean、95% CI 或 SD；
- 明确区分 training seeds、evaluation trials 和 episodes；
- 如果暂时无法补训，应将 ablation 和 league causal claims 标记为 exploratory。

### P1.10 用 fixed test suite 强化 Table II

项目已有 fixed-suite summary，覆盖：

- 4 scripted tests
- 10 learned-hand tests

这比当前单一 SPC、单一 H1 和单一 mouse operator 更能支撑 robustness claim。现有 summary 中，league 在 4 个 scripted tests 上的 mean/worst TIZ 为 0.7206/0.4952，相比 scripted-only 的 0.5410/0.3702 和 single-hand 的 0.3999/0.1068 更高；在 H1–H10 learned-hand tests 上，league 为 0.2372/0.1658，相比 scripted-only 的 0.1462/0.1166 和 single-hand 的 0.1630/0.1204 更高。

**Required action:** 对 fixed suite 做 repeated-seed evaluation，并将 mean、worst、CVaR 或完整分布用于主表；明确当前 fixed-suite SD 是跨 test conditions 还是跨 seeded trials；修正 `single_h10` label 指向 H1 checkpoint 的命名矛盾；manual mouse 应作为独立 human-in-the-loop stress test，不要与自动条件混合解释。

## 5. Priority 2 — Should Fix

### P2.1 将 ZPD 改写为 ZPD-inspired engineering proxy

Euclidean distance band 可以作为 geometric challenge variable，但并不直接测量 perceived challenge、fatigue、movement quality、motivation 或 therapeutic appropriateness。

建议统一使用：

- “ZPD-inspired distance band”
- “distance-based challenge band”
- “platform-specific operating band”

除非后续有 patient/therapist calibration 证据，否则不应把该距离环带写成临床意义上的 ZPD。

### P2.2 补充 ethics、consent 和 image consent

明确说明：

- healthy pilot 是否获得 IRB/ethics approval、exemption 或 institutional determination
- manual mouse operator 是否属于 human participant data
- data/video/image publication consent
- safety protocol and stopping rules

### P2.3 扩展 physical systems reporting

报告：

- camera resolution and update rate
- YOLO/MediaPipe rate and accuracy
- homography/calibration error
- policy inference latency
- command/control rate
- dropout definition and maximum dead-reckoning duration
- command semantics and clipping
- microrobot tracking error/slip
- magnet and actuator limits
- raw and smoothed signals with filter definition

“zero-shot”应写成 **zero-shot policy deployment after perception/workspace calibration**，而不是整个 system stack 的 zero-shot transfer。

### P2.4 增加 safety/failure-mode table

至少覆盖：

- prolonged hand-tracking loss
- microrobot detection loss
- workspace boundary approach
- command saturation
- magnetic decoupling/slip
- catch event
- emergency stop and controlled shutdown

### P2.5 明确 rehabilitation task scope

当前任务更接近 visuomotor pursuit、reaching-like motion、tracking、coordination 和 continuous correction。它不等同于 grasping、ADL、strength training 或 functional recovery。

说明 therapist 如何调节：

- band center/width
- target speed
- workspace region
- direction
- rest breaks and session duration
- fatigue or compensatory movement criteria

### P2.6 规范 domain terminology

需要审慎处理：

- “neural delay” → sensorimotor/response delay proxy
- “cognitive factor” → pursuit strategy or intent policy
- “engagement” → active participation unless measured
- “real-world rehabilitation platform” → physical proof-of-concept platform
- “microrobot”需给出 size、mass、material 和 magnetic properties；若尺度不符合，应考虑 “miniature magnetic robot”

### P2.7 外部核验近期 references

2025–2026 references、YOLOv12 publication details 和近期 systematic reviews 应逐条核验。由于本次 review 未完成完整的外部 bibliography fact-check，不能把参考文献存在性视为已经确认。

## 6. Priority 3 — Polish

1. 统一 **TIZ**，修正 Fig. 3/Fig. 4 中的 “TIS”。
2. Fig. 2 明确真实 temporal input dimensions；修正 horizon label 与 H=8 的一致性。
3. Fig. 4(d) 给出 raw values、sample sizes 和 uncertainty，而非只给 normalized bars。
4. Fig. 5 说明 smoothing method，并同时提供 raw traces。
5. Fig. 1 减少 decorative shadows/black negative space，使其更符合 IEEE journal figure style。
6. Table I 的 “Range” 改为 “Value/Range”，并标注 units、timestep 和 frame rate。
7. Table II 解释 mouse SD 的计算单位。
8. 在技术问题修复后，再进行 title、abstract 和 conclusion 的最终语言收缩。

## 7. Must rerun vs truthful rewriting / reanalysis

### 必须重训或重跑的情况

- 若坚持把 44-D、16×2 relative displacement 和 trajectory auxiliary 作为 central framework，必须用该架构重新完成核心 league evaluation，并保证 deployment-linked policy 与之对应。
- 若要声称 final 10-generation no-aux league policy 完成了统一 physical zero-shot deployment，必须用单一明确 checkpoint 和固定控制设置重新采集 rollouts。
- 若要把 league 优势解释为 league diversity/PFSP 的因果效果，必须增加约 23M-step budget-matched baselines，并至少比较 uniform pool sampling 与 prioritized sampling。
- 若要把 league 和 ablation 写成 confirmatory method evidence，必须增加 independent training seeds；仅增加 evaluation episodes 不能替代训练重复。
- 若继续保留对 fixed-speed、distance-reactive 和 rule/potential-field controller 的否定或 superiority claim，必须在相同 fixed test suite 下补充这些 baseline evaluations。

### 可通过忠实重写、重新归类或重分析修复的情况

- 将 central simulation story 改写为实际使用的 140-D、16-step、8-channel interaction-history GRU，明确 final league 为 `aux_mode=none`。
- 将 trajectory/multi-risk auxiliary 作为独立 ablation family，不再把 no-aux league result 归因于 auxiliary objective。
- 将 mixed-checkpoint physical rollouts 作为独立 feasibility family，按 checkpoint、stride 和 control rate 分组，并报告现有 aggregate metrics。
- 从 exact artifacts 重新生成 Table II、fixed-suite table、Fig. 3/Fig. 4 数值和 uncertainty labels。
- 将 ZPD 统一改写为 platform-specific distance-based challenge band，并补充真实 ethics、consent、system parameters 和 safety/failure reporting。

## 8. Reviewer summaries

### EIC

认为论文具有清晰问题、独特平台和较好的 Discussion 边界控制，但要求补齐 reproducibility、simple baselines、aggregate physical results，并使 contribution hierarchy 与证据一致。

### Methodology Reviewer

给出最严重的 repository-audited findings：44-D/140-D observation mismatch、no-aux/multi-risk/trajectory-aux mismatch、mixed physical checkpoints，以及已核实的约 23M-vs-5.013M training-budget confound。这些问题在解决前构成 submission blocker。

### Domain Reviewer

认可 proof-of-concept rehabilitation robotics 定位，但要求把 ZPD 改写为 engineering proxy，补充 ethics/consent、task scope 和 microrobot specifications，并避免将 simulated agents 称为 patients。

### Systems Reviewer

要求报告完整 physical stack、latency、tracking/dropout、command semantics、actuation limits、safety failures 和全部 rollouts。当前 physical evidence 只证明 online execution，不证明 robust superiority。

### Devil’s Advocate

指出 TIZ 与 reward objective 之间可能存在 circularity，simple controllers 可能达到类似效果，league gain 可能来自训练预算或数据分布，physical response 也可能来自 smoothing、limits 或 human adaptation。

## 9. Submission-readiness verdict

**Major Revision — not ready for submission.**

最优先的三项工作：

1. 完成所有 claim/figure/table/rollout 的 provenance table。
2. 决定统一重跑 44-D trajectory-aux framework，还是按实际 140-D experiment families 重写全文。
3. 校正 training budget，并完成 budget-matched simple-controller baselines 与 uncertainty reporting。

完成以上三项后，才值得继续做 title、abstract、figures 和 IEEE language polish。

## 10. Review-scope notes

- DOCX manuscript was reviewed read-only; the original file was not modified.
- Document size: 14 pages, approximately 8,425 words, 5 figures, and 3 tables/algorithm blocks.
- Internal Word cross-reference targets were checked; no missing bookmark targets or visible “Error! Reference source not found” markers were detected.
- Full external verification of the 52 bibliography entries was not completed; recent references should be checked separately.
