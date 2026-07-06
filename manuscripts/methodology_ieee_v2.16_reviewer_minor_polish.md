# methodology_ieee_v2.16_reviewer_minor_polish

Complete Markdown review draft revised from `methodology_ieee_v2.15_reviewer_polished.md` after the second independent prose review. This version applies the remaining minor edits on citation placeholders, terminology alignment, paragraph flow, Table II clarity, and auxiliary-head wording.

## III. SYSTEM OVERVIEW AND PROBLEM FORMULATION

We introduce a magnetically actuated, non-contact rehabilitation system that preserves an active motor-training task while avoiding rigid physical coupling between the robot and the patient. The control problem is formulated as a reinforcement learning (RL) task in which the robot regulates interaction difficulty through the motion of a magnetic microrobot.

### A. Hardware Platform

The platform separates the robotic actuator from the patient through a three-layer layout. In the actuation layer, a UR10 robotic arm is placed beneath the workspace, with a permanent magnet mounted on the end-effector to drive the microrobot above. In the interaction layer, a 5-mm acrylic sheet supports the unpowered magnetic microrobot and forms a physical barrier between the arm and the patient. In the perception layer, an overhead RGB camera sends images to a host computer, which estimates the microrobot and hand positions and transmits target pose commands to the UR10 over Ethernet.

The same physical layout serves two design goals: safety and reproducibility. The acrylic interface reduces the risk of blunt impact, excessive traction, and crushing injuries that may occur in direct-contact rehabilitation devices. At the same time, the hardware requires only a collaborative robot, a camera, and a permanent magnet; it does not rely on wearable components, force sensors, onboard batteries, or custom actuators. Passive magnetic actuation keeps the patient-side device lightweight and unpowered, which simplifies setup while limiting the consequences of controller errors during development.

### B. Problem Formulation

We formulate the rehabilitation task as a pursuit-evasion game on a flat workspace. The patient sits in front of the acrylic surface and uses one hand to chase the magnetic microrobot, which is controlled by the robot arm below. The goal is to bring the hand within a small contact radius of the microrobot. Unlike static reaching exercises, the microrobot actively evades the patient's hand, so the task requires sustained visuomotor coordination rather than isolated point-to-point reaching [1].

To keep the task challenging but achievable, we define a Zone of Proximal Development (ZPD) as a target distance range between the hand and the microrobot [2]. The robot should neither escape completely nor yield passively; it should keep the interaction inside this range so that the patient remains engaged without becoming fatigued or frustrated. Patient speed, strategy, delay, and motor noise can all change during interaction, making fixed difficulty settings inadequate.

Conventional controllers are not well suited to this setting. Methods based on artificial potential fields or rule-based impedance control usually rely on hand-crafted responses to instantaneous state feedback. They have limited ability to anticipate higher-level patient strategies, such as interception, or to adapt to heterogeneous movement patterns. Performance can degrade further under tremor, delayed response, or bradykinesia because purposeful movement and involuntary perturbation are not explicitly separated.

Reinforcement learning is useful here because the controller can optimize a long-horizon ZPD objective under variable patient behavior. In our formulation, the robot learns an evasion policy that is rewarded for keeping the hand-microrobot distance within the therapeutic range and penalized for unsafe or uninformative behavior, such as escaping too far or being caught too easily. Temporal observations allow the policy to infer patient intent from movement history. The methodology combines domain randomization, temporal encoding, and league training: the first broadens the range of patient behaviors, the second makes recent motion observable to the policy, and the third prevents over-specialization to a single hand model.

## IV. METHODOLOGY

The training framework is organized around a single requirement: the robot must regulate difficulty under patient behavior that changes in both strategy and motor execution. Cognitive-Motor Decoupled Domain Randomization (CMD-DR) broadens the simulated patient distribution by separating movement intent from biomechanical constraints. A dual-stream encoder then makes patient motion history available to the policy, while an auxiliary future-dynamics head adds a direct learning signal for short-horizon hand-motion prediction. Iterative league training exposes the robot to a growing pool of hand policies so that the learned controller does not depend on one fixed virtual patient.

### A. Cognitive-Motor Decoupled Domain Randomization

A generalizable robot policy must account for variation in both patient strategy and motor execution. CMD-DR addresses these two factors separately. The first layer randomizes the hand strategy by sampling a hand policy from the opponent pool at the beginning of each episode. The pool contains scripted pursuit controllers and trained RL controllers that can intercept or anticipate the microrobot. The selected hand policy produces a desired hand displacement at each timestep. The second layer transforms this desired displacement through motor and sensing constraints. Specifically, the raw displacement is passed through a first-order low-pass filter to model muscle inertia:

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

The temporal buffer h_{t-T:t} contains the T = 16 most recent relative hand displacement vectors, forming a 32-dimensional sequence. From this short history, the encoder can estimate motion direction, response delay, and oscillatory behavior that are unavailable from the current position alone. Relative displacements make the history representation translation-invariant, so the recurrent stream focuses on how the hand is moving rather than where the interaction occurs in the workspace.

The scalar vector and temporal buffer are encoded in separate streams. A Multi-Layer Perceptron (MLP) extracts geometric information from the scalar state, while a Gated Recurrent Unit (GRU) processes the displacement sequence. The two stream outputs are concatenated and passed through a fusion MLP to produce the shared representation used by the policy and value heads.

The ZPD-based reward provides a control signal, but it does not directly supervise how the encoder should represent patient motion. Learning this representation only through policy reward can therefore be slow and unstable. To provide a denser signal, we attach a future-dynamics head to the fused representation. The head predicts the patient's relative displacement over an eight-step horizon and estimates near-catch risk. Because future displacements are already available from the rollout, the auxiliary targets require no additional annotation. The trajectory loss is defined as:

L_traj = E[||D_hat_{t+1:t+H} - D_{t+1:t+H}||_2^2],   H = 8.     (5)

where D_hat_{t+1:t+H} is the predicted future displacement sequence and D_{t+1:t+H} is the observed future displacement sequence. The full training objective combines the PPO loss with the auxiliary trajectory and risk-prediction losses:

L_total = L_PPO + lambda_traj L_traj + lambda_risk L_risk.     (6)

where lambda_traj and lambda_risk control the auxiliary loss weights. Because the future-dynamics head shares the encoder with the policy, its gradients also update the spatial and temporal streams. These auxiliary gradients encourage the encoder to capture patient inertia, delay, and interaction dynamics, which helps the policy anticipate near-future hand motion from recent history.

### C. League Training

Training both policies simultaneously is a straightforward baseline, but it creates a non-stationary learning problem. The return of each agent depends on the policy of the other agent. When both policies are updated together, each update changes the environment faced by the other policy, violating the stationarity assumption of Proximal Policy Optimization (PPO). In practice, this can lead to cyclic adaptation and forgetting: the robot exploits the current hand policy, the hand adapts, and the robot loses strategies that were useful against earlier hands. We avoid this issue by freezing one side of the interaction while training the other. During robot training, the hand policies are fixed and treated as part of the environment. During hand training, the robot policy is fixed.

The objective also differs from purely competitive games. In rehabilitation, the robot should not simply learn to defeat the hand. It should maintain an appropriate level of challenge for patients with different movement abilities. We use hand policy to refer to either a scripted or learned controller that generates hand motion in simulation, and each hand policy in the pool is treated as an opponent during robot training. Because no single hand policy can represent the clinical range of movement behavior, the opponent pool P retains hand policies from different training stages. A pool with both early and later hand policies forces robot updates to encounter simple pursuit behavior as well as more strategic interception patterns.

The full procedure is formalized in Algorithm 1. Each iteration has two phases. In the robot phase, the opponent pool P is fixed, and the robot policy is trained with PPO against hand policies sampled from the pool. In the hand phase, a new hand policy is initialized from the previous hand policy and further trained against the current fixed robot policy. The trained hand policy is then added to P, increasing the diversity of hand behaviors used in the next robot training phase.

```text
Algorithm 1: Iterative League Training
Initialize: opponent pool P <- {pi_H^script}, robot policy pi_R^(0)
for n = 1, 2, ..., N do
    Robot phase: train pi_R^(n) with PPO against opponents sampled from P
    Hand phase: warm-start pi_H^(n) from pi_H^(n-1)
    Fine-tune pi_H^(n) against frozen pi_R^(n)
    P <- P union {pi_H^(n)}
end for
```

Uniform sampling from the opponent pool is inefficient. Weak hand policies provide little learning signal because the robot can maintain the desired distance too easily, whereas very strong hand policies can terminate episodes before useful feedback is obtained. The most informative opponents are challenging but not impossible for the current robot. We therefore use a Prioritized Fictitious Self-Play (PFSP) sampling rule adapted to the ZPD task. Instead of prioritizing opponents by win rate, the rule uses the robot's average normalized episode length against each hand policy as a competitiveness proxy:

q_i = f(w_i),   f(w) = 1 / (1 + exp(-k * (eta - |w - 0.5|))),   P(i) = epsilon + (1 - |P| epsilon) q_i / sum_j q_j.     (7)

where w_i denotes the normalized episode length against opponent i, k controls the sharpness of the priority curve, eta defines the tolerance band around the target value of 0.5, and epsilon is a probability floor. Opponents that produce intermediate episode lengths receive higher sampling probability because they are likely to provide useful training feedback. The floor prevents older hand policies from disappearing from the training distribution, preserving breadth while the sampling mass shifts toward currently informative opponents.

A remaining risk is that the robot may exploit artificial patterns in learned hand policies. Such behavior would reduce the relevance of the learned policy for real rehabilitation. To reduce this risk, the heuristic scripted hand remains in the pool throughout training, keeping the robot exposed to simple, noisy pursuit behavior closer to what may be expected from real patients.

## V. EXPERIMENTS AND RESULTS

The experiments address three questions: whether league training improves robustness across learned hand behaviors, whether temporal motion history is necessary for adaptive difficulty regulation, and whether the learned simulation policy can be connected to the physical UR10 platform through a fixed-rate control pipeline.

### A. Experimental Setup

All simulation experiments use the pursuit-evasion rehabilitation environment described in Section III. The robot controls the magnetic microrobot, and the hand is controlled either by a scripted pursuit controller or by a learned hand policy. The therapeutic objective is defined by a ZPD distance band of 3.5-5.5 workspace units. A step is counted as in-zone when the hand-microrobot distance lies inside this band.

The primary evaluation metric is Time-In-Zone (TIS), the fraction of the episode horizon during which the interaction remains inside the ZPD band. We also report ZPD coverage, episode length, catch rate, and empirical robustness across the learned hand pool. Together, the metrics capture different aspects of the same rehabilitation objective: TIS measures sustained therapeutic interaction over the full horizon, ZPD coverage measures the quality of the realized portion of an episode, episode length captures survival, and catch rate reflects overly easy interactions in which the microrobot is caught too quickly.

We compare three representative robot training protocols. The scripted-only baseline is trained against the stochastic scripted pursuit controller. The single-hand baseline is trained against one learned hand policy. The league policy is trained through iterative opponent-pool expansion, where robot generations are trained against a mixture of scripted and learned hand behaviors. For the network ablation, we compare an MLP policy, a GRU policy with temporal interaction history, and a GRU+Aux policy with the auxiliary future-dynamics head.

### B. League Training and Robustness Evaluation

The league evaluation tests whether opponent-pool training reduces the brittle specialization that can occur when the robot is optimized against a single hand model. The robot is trained for ten generations while the hand pool is expanded with learned hand policies. Opponent identity is not provided to the robot; hand behavior must be inferred from the spatial state and recent displacement history, as would be required when patient capability is observed through movement rather than given as a label.

Fig. 1 shows that league training changes both average behavior and difficult-case behavior. The cross-iteration validation matrix exposes the full robot-hand interaction pattern: early learned hands remain comparatively easy to regulate, whereas later learned hands introduce pursuit strategies that are harder for narrow training regimes. Later robot generations improve across a broader portion of this matrix, indicating that the league does not merely produce a policy that performs well on the final hand policy.

The aggregate robustness metrics make the same point from the lower tail of the learned-hand distribution. Mean TIS improves across generations, but the more relevant result for rehabilitation is that worst-hand TIS and CVaR20 also improve. A controller that raises only the average score could still fail for patients with more difficult movement patterns; the frontier view instead shows later robot generations moving toward policies that retain useful ZPD regulation even on harder hand policies.

The final-generation analysis clarifies the remaining failure modes. Some hand policies still create episodes that become too close, while others push the robot toward too-far interactions. Separating these modes matters because both reduce therapeutic value for different reasons: the former makes the task too easy, and the latter makes it discouraging or unreachable. The league policy maintains useful ZPD coverage despite these different failure patterns, suggesting that the learned behavior is balanced rather than tuned to one type of opponent.

The PFSP sampling traces explain how this balance emerges during training. Sampling probability shifts toward hand policies that remain informative for the current robot generation, while the probability floor keeps earlier hand policies available. Training pressure is reallocated as the league evolves: difficult opponents receive more attention, but older pursuit behaviors are not forgotten. This mechanism links the robustness gains in evaluation to the curriculum induced by the opponent pool.

**Fig. 1. League-training validation and opponent-sampling dynamics for the no-opponent-ID ZPD 3.5-5.5 simulation. (a) Cross-iteration TIS matrix between robot generations and learned hand generations. (b) Mean, worst-hand, and CVaR20 TIS across robot generations. (c) Robustness frontier relating mean TIS, worst-hand TIS, and CVaR20. (d) Final-generation failure decomposition showing too-close rate, too-far rate, and ZPD coverage across test hands. (e) PFSP sampling-probability snapshots during selected training iterations.**

Policy comparison across hand-controller types provides a stricter test of specialization. The final scripted-only, single-hand, and league-trained robot policies are evaluated against both the stochastic scripted pursuit controller and a learned hand policy. Mouse-controlled human-in-the-loop testing is kept separate because it introduces human reaction time and voluntary strategy rather than another automated controller.

**TABLE II: POLICY COMPARISON AGAINST SCRIPTED AND LEARNED HAND POLICIES**

| Robot policy | Scripted-hand TIS | Scripted-hand ZPD | Scripted-hand length | Scripted-hand catch | Learned-hand TIS | Learned-hand ZPD | Learned-hand length | Learned-hand catch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scripted-only | 0.31 | 0.54 | 52 | 60% | 0.19 | 0.54 | 31 | 90% |
| Single-hand | 0.11 | 0.41 | 25 | 95% | 0.49 | 0.60 | 70 | 38% |
| League | 0.38 | 0.55 | 67 | 43% | 0.48 | 0.65 | 64 | 48% |

Table II highlights the cost of training against a narrow hand model. The scripted-only robot remains competitive on the scripted controller but is caught frequently by the learned hand policy, indicating that rule-based pursuit during training does not cover learned strategic behavior. The single-hand robot shows the reverse failure: it performs well against the learned hand policy but degrades sharply on the scripted controller. The league-trained robot is not the best on every scalar metric, but it avoids the severe cross-condition collapse seen in the baselines. For adaptive rehabilitation, that balance is the desired outcome because patient behavior can shift across sessions and even within a single episode.

### C. Network Ablation and Auxiliary Dynamics Analysis

The ablation study turns from opponent diversity to policy representation. The key question is whether the robot can regulate difficulty from the current geometric state alone, or whether it needs recent patient-motion history. The MLP baseline observes the current geometric state but does not explicitly encode recent interaction history. The GRU policy processes the 16-frame relative-displacement buffer, allowing it to infer hand velocity, response delay, and pursuit tendency. The GRU+Aux policy further adds the future-dynamics prediction head described in Section IV.B.

Fig. 2 supports a clear representation-level conclusion: the main performance gain appears when temporal interaction history is introduced. Both recurrent policies learn longer and more rewarding interactions than the MLP baseline, indicating that the robot benefits from observing how the hand has been moving. In this run, the GRU-only and GRU+Aux policies follow similar aggregate learning trends, so temporal sequence modeling is the dominant contributor to control performance.

The auxiliary prediction examples provide a complementary diagnostic of the learned temporal representation. The future-dynamics head captures short-horizon movement direction and diverges gradually at longer horizons, consistent with the uncertainty of multi-step prediction in an interactive task. In this experiment, the auxiliary task encourages the recurrent encoder to represent patient-motion structure and makes that representation inspectable through prediction examples.

**Fig. 2. GRU ablation and auxiliary future-motion prediction. Panels (a) and (b) show smoothed training reward and episode length for MLP, GRU, and GRU+Aux policies trained against the mixed learned-hand and scripted-hand pool. Panels (c) and (d) visualize representative auxiliary future-prediction examples and horizon-dependent trajectory error.**

### D. Real-World Deployment Pipeline

The deployment experiment connects the simulation-trained policy to the physical UR10 platform. It should be interpreted as an implementation bridge rather than a paired sim-to-real benchmark. Real hand kinematics, camera latency, magnetic actuation, and safety constraints are not fully represented in the virtual training environment, so the purpose here is to show how the learned policy can run inside a real-time perception-control stack.

The physical implementation uses the same 44-dimensional observation structure as simulation: robot position from Real-Time Data Exchange (RTDE), hand and microrobot positions from the overhead camera, boundary features, previous action, and the displacement-history buffer. Perception and control run in separate threads to avoid jitter. The camera runs at 10-15 Hz, whereas the robot requires deterministic commands at a fixed control rate. If both processes ran in one thread, a slow You Only Look Once (YOLO) inference step could stall the control loop.

To maintain the 20 Hz control loop, the system uses two threads connected by a single-element message queue. The vision thread captures frames, runs undistortion and detection, and pushes results to the queue. If the queue is full, stale results are replaced. The control thread attempts a non-blocking read each cycle. If a new frame is available, the hand position is updated; otherwise, the system uses dead reckoning based on a low-pass filtered velocity estimate from recent vision updates. The resulting observation is passed to the PPO policy, and the two-dimensional action is mapped through the inverse homography before being sent to the UR10 through the servoL interface.
