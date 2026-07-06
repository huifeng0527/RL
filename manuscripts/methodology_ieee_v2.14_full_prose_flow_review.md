# methodology_ieee_v2.14_full_prose_flow_review

Complete Markdown review draft generated from `methodology_ieee_v2.12_fig1_corrected.docx`. Unchanged sections are preserved; targeted prose-flow revisions are applied inline for review before updating the Word manuscript.

## III. SYSTEM OVERVIEW AND PROBLEM FORMULATION

We introduce a magnetically actuated, non-contact rehabilitation system that preserves an active motor-training task while avoiding rigid physical coupling between the robot and the patient. The control problem is formulated as a reinforcement learning (RL) task in which the robot regulates interaction difficulty through the motion of a magnetic microrobot.

### A. Hardware Platform

The platform separates the robotic actuator from the patient through a three-layer layout. In the actuation layer, a UR10 robotic arm is placed beneath the workspace, with a permanent magnet mounted on the end-effector to drive the microrobot above. In the interaction layer, a 5-mm acrylic sheet supports the unpowered magnetic microrobot and forms a physical barrier between the arm and the patient. In the perception layer, an overhead RGB camera sends images to a host computer, which estimates the microrobot and hand positions and transmits target pose commands to the UR10 over Ethernet.

The same physical layout serves two design goals: safety and reproducibility. The acrylic interface reduces the risk of blunt impact, excessive traction, and crushing injuries that may occur in direct-contact rehabilitation devices. At the same time, the hardware requires only a collaborative robot, a camera, and a permanent magnet; it does not rely on wearable components, force sensors, onboard batteries, or custom actuators. Passive magnetic actuation keeps the patient-side device lightweight and unpowered, which simplifies setup while limiting the consequences of controller errors during development.

### B. Problem Formulation

We formulate the rehabilitation task as a pursuit-evasion game on a flat workspace. The patient sits in front of the acrylic surface and uses one hand to chase the magnetic microrobot, which is controlled by the robot arm below. The goal is to bring the hand within a small contact radius of the microrobot. Unlike static reaching exercises, the microrobot actively evades the patient's hand, requiring continuous visuomotor tracking, trajectory prediction, and real-time motor planning. Goal-directed tracking targets sustained visuomotor coordination rather than isolated point-to-point movement [X].
To keep the task challenging but achievable, we define a Zone of Proximal Development (ZPD) as a target distance range between the hand and the microrobot [X]. The robot should neither escape completely nor yield passively; it should keep the interaction inside this range so that the patient remains engaged without becoming fatigued or frustrated. Patient speed, strategy, delay, and motor noise can all change during interaction, making fixed difficulty settings inadequate.

Conventional controllers are not well suited to this setting. Methods based on artificial potential fields or rule-based impedance control usually rely on hand-crafted responses to instantaneous state feedback. They have limited ability to anticipate higher-level patient strategies, such as interception, or to adapt to heterogeneous movement patterns. Performance can degrade further under tremor, delayed response, or bradykinesia because purposeful movement and involuntary perturbation are not explicitly separated.
Reinforcement learning is useful here because the controller can optimize a long-horizon ZPD objective under variable patient behavior, rather than following a fixed local response rule. In our formulation, the robot learns an evasion policy that is rewarded for keeping the hand-microrobot distance within the therapeutic range and penalized for unsafe or uninformative behavior, such as escaping too far or being caught too easily. Temporal observations allow the policy to infer patient intent from movement history instead of reacting only to instantaneous position error. Section IV introduces domain randomization, temporal encoding, and league training as the three mechanisms used to handle patient variability.

## IV. METHODOLOGY

The training framework combines three components. Cognitive-Motor Decoupled Domain Randomization (CMD-DR) separates patient movement strategy from motor execution limitations. A dual-stream encoder extracts geometric and temporal motion information from noisy observations, and an auxiliary future-dynamics head provides an additional training signal for the temporal representation. Iterative league training then exposes the robot to a growing pool of hand behaviors.

### A. Cognitive-Motor Decoupled Domain Randomization

A generalizable robot policy must account for variation in both patient strategy and motor execution. CMD-DR addresses these two factors separately. The first layer randomizes the hand strategy by sampling a controller from an opponent pool at the beginning of each episode. The pool contains scripted pursuit controllers and trained RL controllers that can intercept or anticipate the microrobot. The selected controller produces a desired hand displacement at each timestep. The second layer transforms this desired displacement through motor and sensing constraints. Specifically, the raw displacement is passed through a first-order low-pass filter to model muscle inertia:

x_t = alpha x_{t-1} + (1 - alpha) u_t,   alpha in (0, 1).     (1)

where u_t denotes the raw desired displacement and x_t denotes the filtered displacement. Acceleration clipping is then applied to limit abrupt changes in motion:

Delta x_t = clip(x_t - x_{t-1}, -a_max, a_max).     (2)

Gaussian observation noise simulates position-estimation errors from the vision system. A sampled temporal delay models delayed motor response by executing a displacement generated several frames earlier. Table I lists the parameter values and sampled ranges for these constraints. By separating intended movement from motor execution, CMD-DR exposes the robot policy to diverse patient-like behaviors and reduces dependence on any single simulated hand model.

| Parameter | Symbol | Range |
| --- | --- | --- |
| Muscle inertia | alpha | 0.7 |
| Maximum acceleration | a_max | 0.15 |
| Gaussian observation noise | sigma | U(0.01, 0.08) |
| Neural delay | delta | 0-3 frames (0-375 ms) |

### B. Dual-Stream Encoder with Future Dynamics Head

A single frame is often insufficient for inferring patient intent. Observed hand motion may reflect sensing noise, motor delay, tremor-like perturbations, or slow response, rather than deliberate pursuit. We therefore use a dual-stream encoder with an auxiliary prediction task.

The encoder separates geometric context from motion history. At each timestep, the 44-dimensional observation vector is divided into a 12-dimensional scalar vector and a 32-dimensional temporal buffer:

o_t = [s_t ; h_{t-T:t}].     (3)

The scalar vector is defined as:

s_t = [p_R; p_H; d(R,H); b_N, b_S, b_E, b_W; stride; a_{t-1}].     (4)

where p_R and p_H denote the two-dimensional positions of the microrobot and the hand, respectively; d(R,H) is their Euclidean distance; b_N, b_S, b_E, and b_W are the signed distances to the workspace boundaries; stride is the current movement step size; and a_{t-1} is the previous robot action. These variables describe the current geometric state of the interaction, including relative position, boundary distance, and recent action information.

The temporal buffer h_{t-T:t} contains the T = 16 most recent relative hand displacement vectors, forming a 32-dimensional sequence. It captures velocity, acceleration, and repeated movement patterns that are unavailable from the current position alone. Using relative displacements rather than absolute positions makes the history representation translation-invariant and emphasizes movement direction, speed, acceleration, and oscillation.
The scalar vector and temporal buffer are encoded in separate streams. A Multi-Layer Perceptron (MLP) extracts geometric information from the scalar state, while a Gated Recurrent Unit (GRU) processes the displacement sequence. The two stream outputs are concatenated and passed through a fusion MLP to produce the shared representation used by the policy and value heads.

The ZPD-based reward provides a control signal, but it does not directly supervise how the encoder should represent patient motion. Learning this representation only through policy reward can therefore be slow and unstable. To provide a denser signal, we attach a future-dynamics head to the fused representation. The head predicts the patient's relative displacement over an eight-step horizon and estimates near-catch risk. Because future displacements are already available from the rollout, the auxiliary targets require no additional annotation. The trajectory loss is defined as:

L_traj = E[||D_hat_{t+1:t+H} - D_{t+1:t+H}||_2^2],   H = 8.     (5)

where D_hat_{t+1:t+H} is the predicted future displacement sequence and D_{t+1:t+H} is the observed future displacement sequence. The full training objective combines the PPO loss with the auxiliary trajectory and risk-prediction losses:

L_total = L_PPO + lambda_traj L_traj + lambda_risk L_risk.     (6)

where lambda_traj and lambda_risk control the auxiliary loss weights. Because the future-dynamics head shares the encoder with the policy, its gradients also update the spatial and temporal streams. Shared auxiliary gradients encourage the encoder to capture patient inertia, delay, and interaction dynamics, enabling the policy to anticipate near-future hand motion rather than reacting only to the current hand-microrobot distance.

### C. League Training

Training both policies simultaneously is a straightforward baseline, but it creates a non-stationary learning problem. The return of each agent depends on the policy of the other agent. When both policies are updated together, each update changes the environment faced by the other policy, violating the stationarity assumption of Proximal Policy Optimization (PPO). In practice, this can lead to cyclic adaptation and forgetting: the robot exploits the current hand policy, the hand adapts, and the robot loses strategies that were useful against earlier hands. We avoid this issue by freezing one side of the interaction while training the other. During robot training, the hand policies are fixed and treated as part of the environment. During hand training, the robot policy is fixed.

The objective also differs from purely competitive games. In rehabilitation, the robot should not simply learn to defeat the hand. It should maintain an appropriate level of challenge for patients with different movement abilities. A single hand policy cannot represent this range of behavior, so we maintain an opponent pool P containing hand policies from different training stages. This pool provides a diverse set of pursuit behaviors for robot training.

The full procedure is formalized in Algorithm 1. Each iteration has two phases. In the robot phase, the opponent pool P is fixed, and the robot policy is trained with PPO against hand policies sampled from the pool. In the hand phase, a new hand policy is initialized from the previous hand policy and further trained against the current fixed robot policy. The trained hand policy is then added to P, increasing the diversity of hand behaviors used in the next robot training phase.

Algorithm 1: Iterative League Training

Initialize: opponent pool P <- {pi_H^script}, robot policy pi_R^(0)

for n = 1, 2, ..., N do

// Robot phase (pool frozen)

Train pi_R^(n) via PPO against opponents sampled

from P using prioritized distribution P(i)

// Hand phase (robot frozen)

Warm-start pi_H^(n) from pi_H^(n-1), fine-tune

against frozen pi_R^(n)

P <- P union {pi_H^(n)}

end for

Uniform sampling from the opponent pool is inefficient. Weak hand policies provide little learning signal because the robot can maintain the desired distance too easily, whereas very strong hand policies can terminate episodes before useful feedback is obtained. The most informative opponents are challenging but not impossible for the current robot. We therefore track the robot's average normalized episode length against each hand policy and sample opponents using a length-prioritized distribution:

P(i) propto f(w_i),  f(w) = 1 / (1 + exp(-k * (eta - |w - 0.5|))).     (7)

where w_i denotes the normalized episode length against opponent i, k controls the sharpness of the distribution, and eta defines the tolerance band around the target value of 0.5. The resulting distribution assigns higher probability to hand policies that produce intermediate episode lengths, concentrating training on informative opponents without requiring a manually specified curriculum.

One possible risk of league training is that the robot may learn to exploit artificial patterns in the simulated hand policies. Such behavior would reduce the relevance of the learned policy for real rehabilitation. To reduce this risk, a fixed probability p_s is always assigned to the heuristic scripted hand policy, keeping the robot exposed to simple, noisy pursuit behavior closer to what may be expected from real patients.

## VI. EXPERIMENTS AND RESULTS

The experiments evaluate three questions: whether league training improves robustness across learned hand behaviors, whether temporal motion history is necessary for adaptive difficulty regulation, and whether the learned simulation policy can be connected to the physical UR10 platform through a fixed-rate control pipeline.

### A. Experimental Setup

All simulation experiments use the pursuit-evasion rehabilitation environment described in Section III. The robot controls the magnetic microrobot, and the hand is controlled either by a scripted pursuit controller or by a learned hand policy. The therapeutic objective is defined by a ZPD distance band of 3.5-5.5 workspace units. A step is counted as in-zone when the hand-microrobot distance lies inside this band.

The primary evaluation metric is Time-In-Zone (TIS), the fraction of the episode horizon during which the interaction remains inside the ZPD band. We also report ZPD coverage, episode length, catch rate, and empirical robustness across the learned hand pool. Together, the metrics capture different aspects of the same rehabilitation objective: TIS measures sustained therapeutic interaction over the full horizon, ZPD coverage measures the quality of the realized portion of an episode, episode length captures survival, and catch rate reflects overly easy interactions in which the microrobot is caught too quickly.

We compare three representative robot training protocols. The scripted-only baseline is trained against the stochastic scripted pursuit controller. The single-agent baseline is trained against one learned hand policy. The league policy is trained through iterative opponent-pool expansion, where robot generations are trained against a mixture of scripted and learned hand behaviors. For the network ablation, we compare an MLP policy, a GRU policy with temporal interaction history, and a GRU+Aux policy with the auxiliary future-dynamics head.

### B. League Training and Robustness Evaluation

The league evaluation tests whether opponent-pool training reduces the brittle specialization that can occur when the robot is optimized against a single hand model. The robot is trained for ten generations while the hand pool is expanded with learned hand policies. Opponent identity is not provided to the robot; hand behavior must be inferred from the spatial state and recent displacement history, as would be required when patient capability is observed through movement rather than given as a label.

**Fig. 1(a) evaluates every robot generation against every learned hand generation, producing a cross-iteration TIS matrix rather than a single final score. The matrix exposes both sides of the league: later robot generations generally become more capable, while later learned hands introduce harder pursuit behaviors. Early learned hands remain easier to regulate, but the harder columns reveal why a robot trained on a narrow opponent set can appear competent while still failing on other hand strategies.**

The aggregate views in Fig. 1(b) and Fig. 1(c) separate average performance from difficult-case robustness. Mean TIS increases across generations, but the more important trend is the improvement in worst-hand TIS and CVaR20, which measure the lower tail of the learned-hand distribution. The robustness frontier makes the same point geometrically: later robot generations move toward policies that improve average ZPD regulation without sacrificing the hardest hand cases.

**Fig. 1(d) and Fig. 1(e) connect final behavior to the training process. The final-generation failure decomposition separates episodes that are too close from those that are too far, showing that different hand generations fail in different ways even when ZPD coverage remains useful. PFSP sampling snapshots show the corresponding training mechanism: sampling probability shifts toward opponents that remain informative, while the probability floor keeps older hand policies in the pool. The league therefore improves robustness not by optimizing one final opponent, but by repeatedly reallocating training pressure across a changing set of hand behaviors.**

**Fig. 1. League-training validation and opponent-sampling dynamics for the no-opponent-ID ZPD 3.5-5.5 simulation. (a) Cross-iteration TIS matrix between robot generations and learned hand generations. (b) Mean, worst-hand, and CVaR20 TIS across robot generations. (c) Robustness frontier relating mean TIS, worst-hand TIS, and CVaR20. (d) Final-generation failure decomposition showing too-close rate, too-far rate, and ZPD coverage across test hands. (e) PFSP sampling-probability snapshots during selected training iterations.**

We further test whether the learned robustness transfers across hand-controller types. For this comparison, the final scripted-only, single-agent, and league-trained robot policies are each evaluated against two hand mechanisms: the stochastic scripted pursuit controller and a learned hand policy. The table entries report TIS, ZPD coverage, episode length, and catch rate in that order. Mouse-controlled human-in-the-loop testing is kept separate because it introduces human reaction time and voluntary strategy rather than another automated controller.

**TABLE II: POLICY COMPARISON ON SCRIPTED AND LEARNED-AGENT HANDS**

| Robot policy | Scripted hand | Agent hand |
| --- | --- | --- |
| Scripted-only | 0.31 / 0.54 / 52 / 60% | 0.19 / 0.54 / 31 / 90% |
| Single-agent | 0.11 / 0.41 / 25 / 95% | 0.49 / 0.60 / 70 / 38% |
| League | 0.38 / 0.55 / 67 / 43% | 0.48 / 0.65 / 64 / 48% |

Table II shows complementary failure modes for the two non-league baselines. The scripted-only robot remains competitive on the scripted controller but is caught frequently by the learned hand policy, indicating that rule-based pursuit during training does not cover learned strategic behavior. The single-agent robot shows the reverse tendency: it performs well against the learned hand policy but degrades on the scripted controller. The league-trained robot does not dominate every metric in every column, but it avoids the severe collapse observed in the baselines and provides the most balanced behavior across both automated test conditions. For adaptive rehabilitation, balanced robustness is more important than overfitting to a single hand model because patient behavior can shift across sessions and within an episode.

### C. Network Ablation and Auxiliary Dynamics Analysis

The ablation study turns from opponent diversity to policy representation. The key question is whether the robot can regulate difficulty from the current geometric state alone, or whether it needs recent patient-motion history. The MLP baseline observes the current geometric state but does not explicitly encode recent interaction history. The GRU policy processes the 16-frame relative-displacement buffer, allowing it to infer hand velocity, response delay, and pursuit tendency. The GRU+Aux policy further adds the future-dynamics prediction head described in Section IV-B.

The upper panels of Fig. 2 show that the main performance gain appears when temporal interaction history is introduced. Both recurrent policies learn longer and more rewarding interactions than the MLP baseline, indicating that the robot benefits from observing how the hand has been moving, not only where it is at the current instant. In this setting, the GRU-only and GRU+Aux policies follow similar aggregate learning trends, so temporal sequence modeling is the dominant contributor to control performance.

The lower panels of Fig. 2 provide a complementary interpretation of the auxiliary head. Rather than treating the auxiliary task as a separate source of reward improvement, we use it to inspect whether the temporal encoder has learned predictive structure in hand motion. The predicted trajectories capture short-horizon movement direction and diverge gradually at longer horizons, consistent with the uncertainty of multi-step prediction in an interactive task. In this run, the auxiliary head mainly serves to encourage and expose a temporal representation of patient dynamics.

**Fig. 2. GRU ablation and auxiliary future-motion prediction. Panels (a) and (b) show smoothed training reward and episode length for MLP, GRU, and GRU+Aux policies trained against the mixed learned-hand and scripted-hand pool. Panels (c) and (d) visualize representative auxiliary future-prediction examples and horizon-dependent trajectory error.**

### D. Real-World Deployment Pipeline

The final part of the experiment section describes how the simulation policy is connected to the physical UR10 platform. The pipeline is best interpreted as an implementation bridge rather than a paired sim-to-real benchmark, because real hand kinematics, camera latency, magnetic actuation, and safety constraints are not fully represented in the virtual training environment.

The physical implementation uses the same 44-dimensional observation structure as simulation: robot position from Real-Time Data Exchange (RTDE), hand and microrobot positions from the overhead camera, boundary features, previous action, and the displacement-history buffer. Perception and control run in separate threads to avoid jitter. The camera runs at 10-15 Hz, whereas the robot requires deterministic commands at a fixed control rate. If both processes ran in one thread, a slow You Only Look Once (YOLO) inference step could stall the control loop.

The system therefore uses two threads connected by a single-element message queue. The vision thread captures frames, runs undistortion and detection, and pushes results to the queue. If the queue is full, stale results are replaced. The control thread runs at 20 Hz and attempts a non-blocking read each cycle. If a new frame is available, the hand position is updated; otherwise, the system uses dead reckoning based on a low-pass filtered velocity estimate from recent vision updates. The resulting observation is passed to the PPO policy, and the two-dimensional action is mapped through the inverse homography before being sent to the UR10 through the servoL interface.
