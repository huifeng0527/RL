# methodology_ieee_v2.25_three_testsets_table

Complete Markdown review draft revised from `methodology_ieee_v2.24_mouse_hand_results.md`. This version merges the scripted-hand, learned-hand, and mouse-hand evaluations into one three-test-set policy comparison table.

## II. SYSTEM OVERVIEW AND PROBLEM FORMULATION

We introduce a magnetically actuated, non-contact rehabilitation system for active upper-limb motor training. The patient interacts with an unpowered magnetic microrobot on a tabletop workspace, while a robotic arm drives the microrobot indirectly through a permanent magnet placed below the surface. This layout preserves an active pursuit task for the patient while avoiding rigid physical coupling between the robot and the human body. We formulate the control problem as a reinforcement learning (RL) task in which the robot regulates interaction difficulty through the motion of the magnetic microrobot.

### A. Hardware Platform

The platform separates actuation, interaction, and perception into three physical layers. In the actuation layer, a UR10 robotic arm is placed beneath the workspace, and a permanent magnet is mounted on its end-effector to drive the microrobot above. In the interaction layer, a 5-mm acrylic sheet supports the unpowered magnetic microrobot and forms a physical barrier between the robotic arm and the patient. In the perception layer, an overhead RGB camera sends images to a host computer, which estimates the positions of the microrobot and the hand and transmits target pose commands to the UR10 over Ethernet.

This layout supports both safety and reproducibility. The acrylic interface reduces risks associated with direct-contact devices, including impact, excessive traction, and local compression. The hardware also requires only a collaborative robot, a camera, and a permanent magnet. It does not rely on wearable components, force sensors, onboard batteries, or custom actuators. Passive magnetic actuation keeps the patient-side device lightweight and unpowered, which simplifies setup and limits the consequences of controller errors during development.

### B. Problem Formulation

We formulate the rehabilitation task as a pursuit-evasion interaction on a flat workspace. The patient sits in front of the acrylic surface and uses one hand to chase the magnetic microrobot, which is controlled by the robot arm below. The task goal is to bring the hand within a small contact radius of the microrobot. Unlike static reaching exercises, the microrobot actively evades the patient's hand. The task therefore requires sustained visuomotor coordination rather than isolated point-to-point reaching [1].

To keep the task challenging but achievable, we define a Zone of Proximal Development (ZPD) as a target distance range between the hand and the microrobot [2]. The robot should neither escape completely nor yield passively. Instead, it should keep the interaction inside this range so that the patient remains engaged without becoming fatigued or frustrated. Patient speed, pursuit strategy, response delay, and motor noise can all change during interaction, making fixed difficulty settings inadequate.

Moreover, conventional controllers are limited in this setting because methods such as artificial potential fields or rule-based impedance control usually rely on hand-crafted responses to instantaneous state feedback. These methods struggle to anticipate strategic pursuit behaviors such as interception or to adapt to heterogeneous movement patterns. Their performance can also degrade under tremor, delayed response, or bradykinesia, since purposeful movement and involuntary perturbation are not explicitly separated. These limitations motivate an RL formulation in which the controller optimizes a long-horizon ZPD objective under variable patient behavior rather than reacting only to the current hand-microrobot distance. In our formulation, the robot learns an evasion policy that is rewarded for keeping the interaction within the therapeutic range and penalized for unsafe or uninformative behavior, such as escaping too far or being caught too easily. The key challenge is to learn this policy without assuming a fixed or fully observable patient response model.

## IV. METHODOLOGY

The robot must therefore regulate difficulty while patient behavior changes in both strategy and motor execution. Our approach treats this variability as a simulation problem, an observation problem, and a training-distribution problem. Cognitive-Motor Decoupled Domain Randomization (CMD-DR) broadens the simulated patient distribution by separating movement intent from biomechanical constraints. A dual-stream encoder makes patient motion history available to the policy, while an auxiliary future-dynamics head adds a direct learning signal for short-horizon hand-motion prediction. Iterative league training exposes the robot to a growing pool of hand policies so that the learned controller does not depend on one fixed virtual patient.

### A. Cognitive-Motor Decoupled Domain Randomization

In rehabilitation settings, patients can differ both in how they choose to move and in how accurately they can execute those movements, making it difficult for a single model to capture all variations. This approach separates these two sources of variability by generating intended hand motion first and then applying motor-execution constraints, along with observation noise and temporal delay to model sensing uncertainty. At the strategy level, a hand policy sampled from a pool of scripted and learned controllers determines the intended displacement at each timestep. Scripted controllers provide direct pursuit behavior, while trained RL controllers can produce more anticipatory strategies such as interception. At the execution level, this intended displacement is transformed by constraints that approximate patient motor dynamics and sensing uncertainty.

The raw desired displacement is first passed through a first-order low-pass filter to model muscle inertia, as shown in (1).

x_t = alpha x_{t-1} + (1 - alpha) u_t, alpha in (0, 1). (1)

where u_t denotes the raw desired displacement and x_t denotes the filtered displacement. Acceleration clipping then limits abrupt changes in motion, as shown in (2).

Delta x_t = clip(x_t - x_{t-1}, -a_max, a_max). (2)

Gaussian observation noise simulates position-estimation errors from the vision system. A sampled temporal delay models delayed motor response by executing a displacement generated several frames earlier. Table I lists the values and sampled ranges used for these constraints. Through this decoupled design, the robot policy is exposed to diverse patient-like behaviors without tying training to a single simulated hand model.

| Parameter | Symbol | Range |
| --- | --- | --- |
| Muscle inertia | alpha | 0.7 |
| Maximum acceleration | a_max | 0.15 |
| Gaussian observation noise | sigma | U(0.01, 0.08) |
| Neural delay | delta | 0-3 frames (0-375 ms) |

### B. Dual-Stream Encoder with Future Dynamics Head

the proposed domain randomization scheme exposes the robot to diverse interaction behaviors during training, but the robot does not observe the underlying behavior model directly. It must infer behavior patterns from the current interaction state and recent hand motion. The current hand position alone cannot distinguish deliberate pursuit from sensing noise, motor delay, tremor-like perturbations, or slow response. We therefore encode the observation with two streams, one for instantaneous geometric context and one for recent hand-motion history.

At each timestep, the 44-dimensional observation vector is divided into a 12-dimensional scalar vector and a 32-dimensional temporal buffer, as shown in (3).

o_t = [s_t ; h_{t-T}]. (3)

The scalar component represents the instantaneous geometry of the interaction and is defined in (4).

s_t = [p_R; p_H; d(R,H); b_N, b_S, b_E, b_W; stride; a_{t-1}]. (4)

Here, p_R and p_H denote the two-dimensional positions of the microrobot and the hand. d(R,H) is their Euclidean distance. b_N, b_S, b_E, and b_W denote signed distances to the workspace boundaries. The variables stride and a_{t-1} represent the current movement step size and the previous robot action, respectively.

The temporal buffer h_{t-T} contains the T = 16 most recent relative hand displacement vectors, forming a 32-dimensional sequence. This history allows the encoder to estimate motion direction, response delay, and oscillatory behavior that are unavailable from the current position alone. Relative displacements make the history representation translation-invariant, so the recurrent stream focuses on how the hand is moving rather than where the interaction occurs in the workspace.

The scalar vector and temporal buffer are encoded in separate streams. An MLP extracts geometric information from the scalar state, while a GRU processes the displacement sequence. The two stream outputs are concatenated and passed through a fusion MLP to produce the shared representation used by the policy and value heads.

The ZPD-based reward provides the main control signal, but it only supervises the encoder indirectly through policy optimization. To provide a denser learning signal for motion representation, we attach a future-dynamics head to the fused representation. The head predicts the relative hand displacement over an eight-step horizon and estimates near-catch risk. Because future displacements and near-catch labels are available from the rollout, these auxiliary targets require no additional annotation.

The trajectory prediction loss is given in (5).

L_traj = E[||D_hat_{t+1+H} - D_{t+1+H}||_2^2], H = 8. (5)

where D_hat_{t+1+H} is the predicted future displacement sequence and D_{t+1+H} is the observed future displacement sequence. The risk prediction loss L_risk supervises the near-catch estimate. The full training objective combines the PPO loss with the auxiliary trajectory and risk-prediction losses in (6).

L_total = L_PPO + lambda_traj L_traj + lambda_risk L_risk. (6)

where lambda_traj and lambda_risk control the auxiliary loss weights. Because the future-dynamics head shares the encoder with the policy, its gradients also update the spatial and temporal streams. These auxiliary gradients encourage the encoder to capture motion inertia, delay, and interaction dynamics, helping the policy anticipate near-future hand motion from recent history.

### C. League Training

Even with a well-designed motion representation, achieving stable and adaptive difficulty regulation remains challenging when the training distribution lacks diversity in hand behaviors. A robot trained against a single hand policy may over-specialize to that policy and fail when the pursuit strategy changes. At the same time, training the robot and hand policies simultaneously creates a non-stationary learning problem. The actions of both agents jointly determine the next environment state, so each agent’s transition dynamics and resulting returns are directly influenced by the current policy of the other agent. When both policies are updated together, each update changes the environment faced by the other policy, violating the stationarity assumption of Proximal Policy Optimization (PPO). In practice, the interaction can enter a cycle in which the robot exploits the current hand policy, the hand adapts, and the robot loses strategies that were useful against earlier hands.

We address these two issues separately. To reduce non-stationarity, one side of the interaction is frozen while the other is trained. During robot training, the hand policies are fixed and treated as part of the environment. During hand training, the robot policy is fixed. To reduce over-specialization, the robot is trained against an opponent pool P that retains hand policies from different training stages. Each hand policy is a scripted or learned controller that generates hand motion in simulation. A pool with both early and later hand policies exposes the robot to simple pursuit behavior as well as more strategic interception patterns.

The rehabilitation objective also differs from that of a purely competitive game. The robot should not simply learn to defeat the hand. It should maintain an appropriate level of challenge for patients with different movement abilities. For this reason, the opponent pool is used not to maximize adversarial difficulty, but to broaden the range of hand behaviors under which the robot must maintain the ZPD interaction.

The iterative procedure is formalized in Algorithm 1. Each iteration has two phases. In the robot phase, the opponent pool P is fixed, and the robot policy is trained with PPO against hand policies sampled from the pool. In the hand phase, a new hand policy is initialized from the previous hand policy and further trained against the current fixed robot policy. The trained hand policy is then added to P, increasing the diversity of hand behaviors used in the next robot training phase.

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

Uniform sampling from the opponent pool is inefficient. Weak hand policies provide little learning signal because the robot can maintain the ZPD range too easily, whereas very strong hand policies can terminate episodes before useful feedback is obtained. The most informative opponents are challenging but not impossible for the current robot. We therefore use a Prioritized Fictitious Self-Play (PFSP) sampling rule adapted to the ZPD task. Instead of prioritizing opponents by win rate, the rule uses the robot's average normalized episode length against each hand policy as a competitiveness proxy in (7).

q_i = f(w_i), f(w) = 1 / (1 + exp(-k * (eta - |w - 0.5|))), P(i) = epsilon + (1 - |P| epsilon) q_i / sum_j q_j. (7)

where w_i denotes the normalized episode length against opponent i, k controls the sharpness of the priority curve, eta defines the tolerance band around the target value of 0.5, and epsilon is a probability floor. Opponents that produce intermediate episode lengths receive higher sampling probability because they are likely to provide useful training feedback. The floor prevents older hand policies from disappearing from the training distribution, preserving breadth while the sampling mass shifts toward currently informative opponents.

PFSP may increase the sampling probability of learned hand policies that remain informative for the current robot generation. A remaining risk is that the robot may exploit artificial patterns in these learned policies rather than learn behavior relevant to real rehabilitation. To reduce this risk, the heuristic scripted hand remains in the pool throughout training, keeping the robot exposed to simple, noisy pursuit behavior closer to what may be expected from real patients.

## V. EXPERIMENTS AND RESULTS

The experiments evaluate robustness across learned hand behaviors, the contribution of temporal patient-history encoding, and real-time deployment on the physical UR10 platform.

### A. Experimental Setup

All simulation experiments use the pursuit-evasion rehabilitation environment described in Section III. The robot controls the magnetic microrobot, and the hand is controlled either by a scripted pursuit controller or by a learned hand policy. The therapeutic objective is defined by a ZPD distance band of 3.5-5.5 workspace units. A step is counted as in-zone when the hand-microrobot distance lies inside this band.

The primary evaluation metric is Time-In-Zone (TIS), defined as the fraction of the episode horizon during which the interaction remains inside the ZPD band. We also report ZPD coverage, episode length, catch rate, and empirical robustness across the learned hand pool. These metrics capture complementary aspects of adaptive difficulty regulation. TIS measures sustained therapeutic interaction over the full horizon. ZPD coverage measures the quality of the realized portion of an episode, while episode length captures survival. Catch rate identifies overly easy interactions in which the microrobot is caught too quickly.

The protocol comparison isolates the effect of opponent diversity. The scripted-only baseline is trained against the stochastic scripted pursuit controller, the single-hand baseline against one learned hand policy, and the league policy through iterative opponent-pool expansion. The representation ablation compares an MLP policy, a GRU policy with temporal interaction history, and a GRU+Aux policy with the auxiliary future-dynamics head.

### B. League Training and Robustness Evaluation

The league evaluation tests whether opponent-pool training reduces the brittle specialization that can occur when the robot is optimized against a single hand model. The robot is trained for ten generations while the hand pool is expanded with learned hand policies. 

League training improves both average behavior and difficult-case behavior, as shown in Fig. 1. The cross-iteration validation matrix evaluates each robot generation against each learned hand policy. Early learned hand policies remain comparatively easy to regulate, whereas later policies introduce pursuit strategies that are harder for narrow training regimes. Later robot generations improve across a broader portion of this matrix, indicating that the league does not merely produce a policy that performs well on the final hand policy.

Lower-tail robustness is particularly important for rehabilitation. Mean TIS improves across generations, but worst-hand TIS and CVaR20 also improve. A controller that raises only the average score could still fail for patients with more difficult movement patterns. The frontier view shows that later robot generations move toward policies that retain useful ZPD regulation even on harder hand policies.

The final-generation results clarify the remaining failure modes. Some hand policies still create episodes that become too close, while others push the robot toward too-far interactions. Separating these modes matters because both reduce therapeutic value for different reasons. Too-close interactions make the task too easy, while too-far interactions make the task discouraging or unreachable. The league policy maintains useful ZPD coverage despite these different failure patterns, suggesting that the learned behavior is balanced rather than tuned to one type of opponent.

The PFSP sampling traces link this balanced performance to the curriculum induced by opponent sampling. Sampling probability shifts toward hand policies that remain informative for the current robot generation, while the probability floor keeps earlier hand policies available. As the league evolves, difficult opponents receive more attention without allowing older pursuit behaviors to be forgotten.

**Fig. 1. League-training validation and opponent-sampling dynamics for the no-opponent-ID ZPD 3.5-5.5 simulation. (a) Cross-iteration TIS matrix between robot generations and learned hand generations. (b) Mean, worst-hand, and CVaR20 TIS across robot generations. (c) Robustness frontier relating mean TIS, worst-hand TIS, and CVaR20. (d) Final-generation failure decomposition showing too-close rate, too-far rate, and ZPD coverage across test hands. (e) PFSP sampling-probability snapshots during selected training iterations.**

Policy comparison across three hand test sets provides a stricter test of specialization. The final scripted-only, single-hand, and league-trained robot policies are evaluated against a stochastic scripted pursuit controller, a learned hand policy, and a mouse-controlled hand. The mouse-controlled condition introduces human reaction time and voluntary pursuit strategy, so it is interpreted as a human-in-the-loop stress test rather than another automated opponent.

**TABLE II: POLICY COMPARISON ACROSS THREE HAND TEST SETS**

| Robot policy | Test set | TIS | ZPD coverage | Episode length | Catch rate |
| --- | --- | --- | --- | --- | --- |
| Scripted-only | Scripted hand | 0.31 | 0.54 | 52 | 60% |
| Scripted-only | Learned hand | 0.19 | 0.54 | 31 | 90% |
| Scripted-only | Mouse hand | 0.14 | 0.50 | 33 | 100% |
| Single-hand | Scripted hand | 0.11 | 0.41 | 25 | 95% |
| Single-hand | Learned hand | 0.49 | 0.60 | 70 | 38% |
| Single-hand | Mouse hand | 0.19 | 0.48 | 40 | 95% |
| League | Scripted hand | 0.38 | 0.55 | 67 | 43% |
| League | Learned hand | 0.48 | 0.65 | 64 | 48% |
| League | Mouse hand | 0.49 | 0.66 | 74 | 60% |

Training against a narrow hand model leads to complementary failure modes across the three test sets. The scripted-only robot remains competitive on the scripted hand but is caught frequently by the learned and mouse-controlled hands, indicating that rule-based pursuit during training does not cover strategic or human-controlled pursuit behavior. The single-hand robot shows the reverse failure; it performs well against the learned hand policy but degrades sharply on the scripted and mouse-controlled hands.

The league-trained robot is not the best on every scalar metric, but it avoids the severe cross-condition collapse seen in the baselines. Its strongest relative advantage appears under mouse-controlled pursuit, where it produces longer interactions with higher TIS and ZPD coverage while reducing the catch rate from near saturation to 60%. For adaptive rehabilitation, this balanced performance is more important than optimizing a single test condition, because patient behavior can shift across controllers, sessions, and voluntary strategies. The mouse-controlled result remains a stress test rather than a clinical validation, but it provides additional human-in-the-loop evidence that the league policy is more robust to hand behavior outside a single scripted or learned controller.

### C. Network Ablation and Auxiliary Dynamics Analysis

Opponent diversity alone does not determine whether the robot can use the information available in each observation. The ablation study asks whether the robot can regulate difficulty from the current geometric state alone, or whether it needs recent patient-motion history. The MLP baseline observes the current geometric state but does not explicitly encode recent interaction history. The GRU policy processes the 16-frame relative-displacement buffer, allowing it to infer hand velocity, response delay, and pursuit tendency. The GRU+Aux policy further adds the future-dynamics prediction head described in Section IV.B.

The main performance gain appears when temporal interaction history is introduced, as shown in Fig. 2. Both recurrent policies learn longer and more rewarding interactions than the MLP baseline, indicating that the robot benefits from observing how the hand has been moving. In this run, the GRU-only and GRU+Aux policies follow similar aggregate learning trends, so temporal sequence modeling is the dominant contributor to control performance.

The auxiliary prediction examples provide a complementary diagnostic of the learned temporal representation. The future-dynamics head captures short-horizon movement direction and diverges gradually at longer horizons, consistent with the uncertainty of multi-step prediction in an interactive task. In this experiment, the auxiliary task encourages the recurrent encoder to represent patient-motion structure and makes that representation inspectable through prediction examples.

**Fig. 2. GRU ablation and auxiliary future-motion prediction. Panels (a) and (b) show smoothed training reward and episode length for MLP, GRU, and GRU+Aux policies trained against the mixed learned-hand and scripted-hand pool. Panels (c) and (d) visualize representative auxiliary future-prediction examples and horizon-dependent trajectory error.**

### D. Real-World Deployment Pipeline

The deployment experiment connects the simulation-trained policy to the physical UR10 platform. Real hand kinematics, camera latency, magnetic actuation, and safety constraints are not fully represented in the virtual training environment. The experiment therefore focuses on whether the learned policy can run inside a real-time perception-control stack rather than on a paired sim-to-real benchmark.

The physical implementation uses the same 44-dimensional observation structure as simulation. The observation includes robot position from Real-Time Data Exchange (RTDE), hand and microrobot positions from the overhead camera, boundary features, previous action, and the displacement-history buffer. Perception and control run in separate threads to avoid jitter. The camera runs at 10-15 Hz, whereas the robot requires deterministic commands at a fixed control rate. If both processes ran in one thread, a slow You Only Look Once (YOLO) inference step could stall the control loop.

To maintain the 20 Hz control loop, the system uses two threads connected by a single-element message queue. The vision thread captures frames, runs undistortion and detection, and pushes results to the queue. If the queue is full, stale results are replaced. The control thread attempts a non-blocking read each cycle. If a new frame is available, the hand position is updated; otherwise, the system uses dead reckoning based on a low-pass filtered velocity estimate from recent vision updates. The resulting observation is passed to the PPO policy, and the two-dimensional action is mapped through the inverse homography before being sent to the UR10 through the servoL interface.
