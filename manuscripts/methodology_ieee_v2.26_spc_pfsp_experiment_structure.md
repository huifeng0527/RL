# methodology_ieee_v2.26_spc_pfsp_experiment_structure

Complete Markdown review draft revised from `methodology_ieee_v2.25_three_testsets_table.md`. This version formalizes the scripted pursuit controller, updates the PFSP sampling rule according to the implementation, and restructures the experiments section so simulation and physical deployment protocols are separated.

## II. SYSTEM OVERVIEW AND PROBLEM FORMULATION

We introduce a magnetically actuated, non-contact rehabilitation system for active upper-limb motor training. The patient interacts with an unpowered magnetic microrobot on a tabletop workspace, while a robotic arm drives the microrobot indirectly through a permanent magnet placed below the surface. This layout preserves an active pursuit task for the patient while avoiding rigid physical coupling between the robot and the human body. We formulate the control problem as a reinforcement learning (RL) task in which the robot regulates interaction difficulty through the motion of the magnetic microrobot.

### A. Hardware Platform

The platform separates actuation, interaction, and perception into three physical layers. In the actuation layer, a UR10 robotic arm is placed beneath the workspace, and a permanent magnet is mounted on its end-effector to drive the microrobot above. In the interaction layer, a 5-mm acrylic sheet supports the unpowered magnetic microrobot and forms a physical barrier between the robotic arm and the patient. In the perception layer, an overhead RGB camera sends images to a host computer, which estimates the positions of the microrobot and the hand and transmits target pose commands to the UR10 over Ethernet.

This layout supports both safety and reproducibility. The acrylic interface reduces risks associated with direct-contact devices, including impact, excessive traction, and local compression. The hardware also requires only a collaborative robot, a camera, and a permanent magnet. It does not rely on wearable components, force sensors, onboard batteries, or custom actuators. Passive magnetic actuation keeps the patient-side device lightweight and unpowered, which simplifies setup and limits the consequences of controller errors during development.

### B. Problem Formulation

We formulate the rehabilitation task as a pursuit-evasion interaction on a flat workspace. The patient sits in front of the acrylic surface and uses one hand to chase the magnetic microrobot, which is controlled by the robot arm below. The task goal is to bring the hand within a small contact radius of the microrobot. Unlike static reaching exercises, the microrobot actively evades the patient's hand. The task therefore requires sustained visuomotor coordination rather than isolated point-to-point reaching [1].

To keep the task challenging but achievable, we define a Zone of Proximal Development (ZPD) as a target distance range between the hand and the microrobot [2]. The robot should neither escape completely nor yield passively. Instead, it should keep the interaction inside this range so that the patient remains engaged without becoming fatigued or frustrated. Patient speed, pursuit strategy, response delay, and motor noise can all change during interaction, making fixed difficulty settings inadequate.

Conventional controllers are limited in this setting because methods such as artificial potential fields or rule-based impedance control usually rely on hand-crafted responses to instantaneous state feedback. These methods struggle to anticipate strategic pursuit behaviors such as interception or to adapt to heterogeneous movement patterns. Their performance can also degrade under tremor, delayed response, or bradykinesia, since purposeful movement and involuntary perturbation are not explicitly separated. These limitations motivate an RL formulation in which the controller optimizes a long-horizon ZPD objective under variable patient behavior rather than reacting only to the current hand-microrobot distance. In our formulation, the robot learns an evasion policy that is rewarded for keeping the interaction within the therapeutic range and penalized for unsafe or uninformative behavior, such as escaping too far or being caught too easily. The key challenge is to learn this policy without assuming a fixed or fully observable patient response model.

## III. METHODOLOGY

The robot must therefore regulate difficulty while patient behavior changes in both strategy and motor execution. Our approach treats this variability as a simulation problem, an observation problem, and a training-distribution problem. Cognitive-Motor Decoupled Domain Randomization (CMD-DR) broadens the simulated patient distribution by separating movement intent from biomechanical constraints. A dual-stream encoder makes patient motion history available to the policy, while an auxiliary future-dynamics head adds a direct learning signal for short-horizon hand-motion prediction. Iterative league training exposes the robot to a growing pool of hand policies so that the learned controller does not depend on one fixed virtual patient.

### A. Cognitive-Motor Decoupled Domain Randomization

In rehabilitation settings, patients can differ both in how they choose to move and in how accurately they can execute those movements, making it difficult for a single model to capture all variations. CMD-DR separates these two sources of variability by generating intended hand motion first and then applying motor-execution constraints, observation noise, and temporal delay. At the strategy level, a hand policy sampled from the opponent pool determines the intended displacement at each timestep. The pool contains a stochastic scripted pursuit controller (SPC) and trained RL hand policies. The SPC provides a transparent pursuit model, while learned hand policies can produce more anticipatory strategies such as interception.

The SPC samples a stride rho_ep once per episode and uses it for the whole episode. At each timestep, it either moves toward the microrobot or makes a small random exploratory move. Its intended displacement is defined in (1).

u_t^SPC = rho_ep [ z_t xi_hat_t + (1 - z_t) (p_R,t - p_H,t) / ||p_R,t - p_H,t||_2 ],   z_t ~ Bernoulli(epsilon). (1)

Here, p_R,t and p_H,t are the microrobot and hand positions, xi_hat_t is a normalized random direction, rho_ep is sampled from U(0.45, 0.70), and epsilon = 0.05. The learned hand policies replace (1) with their policy outputs, but the resulting intended displacement is passed through the same motor-execution model.

The raw desired displacement is first passed through a first-order low-pass filter to model muscle inertia, as shown in (2).

x_t = alpha x_{t-1} + (1 - alpha) u_t, alpha in (0, 1). (2)

where u_t denotes the raw desired displacement and x_t denotes the filtered displacement. Acceleration clipping then limits abrupt changes in motion, as shown in (3).

Delta x_t = clip(x_t - x_{t-1}, -a_max, a_max). (3)

Gaussian observation noise simulates position-estimation errors from the vision system. A sampled temporal delay models delayed motor response by executing a displacement generated several frames earlier. Table I lists the values and sampled ranges used for these constraints. Through this decoupled design, the robot policy is exposed to diverse patient-like behaviors without tying training to a single simulated hand model.

| Parameter | Symbol | Range |
| --- | --- | --- |
| Muscle inertia | alpha | 0.7 |
| Maximum acceleration | a_max | 0.15 |
| Gaussian observation noise | sigma | U(0.01, 0.08) |
| Neural delay | delta | 0-3 frames (0-375 ms) |

### B. Dual-Stream Encoder with Future Dynamics Head

The proposed domain randomization scheme exposes the robot to diverse interaction behaviors during training, but the robot does not observe the underlying behavior model directly. It must infer behavior patterns from the current interaction state and recent hand motion. The current hand position alone cannot distinguish deliberate pursuit from sensing noise, motor delay, tremor-like perturbations, or slow response. We therefore encode the observation with two streams, one for instantaneous geometric context and one for recent hand-motion history.

At each timestep, the 44-dimensional observation vector is divided into a 12-dimensional scalar vector and a 32-dimensional temporal buffer, as shown in (4).

o_t = [s_t ; h_{t-T:t}]. (4)

The scalar component represents the instantaneous geometry of the interaction and is defined in (5).

s_t = [p_R; p_H; d(R,H); b_N, b_S, b_E, b_W; stride; a_{t-1}]. (5)

Here, p_R and p_H denote the two-dimensional positions of the microrobot and the hand. d(R,H) is their Euclidean distance. b_N, b_S, b_E, and b_W denote signed distances to the workspace boundaries. The variables stride and a_{t-1} represent the current movement step size and the previous robot action, respectively.

The temporal buffer h_{t-T:t} contains the T = 16 most recent relative hand displacement vectors, forming a 32-dimensional sequence. This history allows the encoder to estimate motion direction, response delay, and oscillatory behavior that are unavailable from the current position alone. Relative displacements make the history representation translation-invariant, so the recurrent stream focuses on how the hand is moving rather than where the interaction occurs in the workspace.

The scalar vector and temporal buffer are encoded in separate streams. An MLP extracts geometric information from the scalar state, while a GRU processes the displacement sequence. The two stream outputs are concatenated and passed through a fusion MLP to produce the shared representation used by the policy and value heads.

The ZPD-based reward provides the main control signal, but it only supervises the encoder indirectly through policy optimization. To provide a denser learning signal for motion representation, we attach a future-dynamics head to the fused representation. The head predicts the relative hand displacement over an eight-step horizon and estimates near-catch risk. Because future displacements and near-catch labels are available from the rollout, these auxiliary targets require no additional annotation.

The trajectory prediction loss is given in (6).

L_traj = E[||D_hat_{t+1:t+H} - D_{t+1:t+H}||_2^2], H = 8. (6)

where D_hat_{t+1:t+H} is the predicted future displacement sequence and D_{t+1:t+H} is the observed future displacement sequence. The risk prediction loss L_risk supervises the near-catch estimate. The full training objective combines the PPO loss with the auxiliary trajectory and risk-prediction losses in (7).

L_total = L_PPO + lambda_traj L_traj + lambda_risk L_risk. (7)

where lambda_traj and lambda_risk control the auxiliary loss weights. Because the future-dynamics head shares the encoder with the policy, its gradients also update the spatial and temporal streams. These auxiliary gradients encourage the encoder to capture motion inertia, delay, and interaction dynamics, helping the policy anticipate near-future hand motion from recent history.

### C. League Training

Even with a well-designed motion representation, achieving stable and adaptive difficulty regulation remains challenging when the training distribution lacks diversity in hand behaviors. A robot trained against a single hand policy may over-specialize to that policy and fail when the pursuit strategy changes. At the same time, training the robot and hand policies simultaneously creates a non-stationary learning problem. The actions of both agents jointly determine the next environment state, so each agent’s transition dynamics and resulting returns are directly influenced by the current policy of the other agent. When both policies are updated together, each update changes the environment faced by the other policy, violating the stationarity assumption of Proximal Policy Optimization (PPO). In practice, the interaction can enter a cycle in which the robot exploits the current hand policy, the hand adapts, and the robot loses strategies that were useful against earlier hands.

We address these two issues separately. To reduce non-stationarity, one side of the interaction is frozen while the other is trained. During robot training, the hand policies are fixed and treated as part of the environment. During hand training, the robot policy is fixed. To reduce over-specialization, the robot is trained against an opponent pool P that retains the SPC and learned hand policies from different training stages. Each hand policy is a scripted or learned controller that generates hand motion in simulation. A pool with the SPC and multiple learned hand policies exposes the robot to direct pursuit behavior as well as more strategic interception patterns.

The rehabilitation objective also differs from that of a purely competitive game. The robot should not simply learn to defeat the hand. It should maintain an appropriate level of challenge for patients with different movement abilities. For this reason, the opponent pool is used not to maximize adversarial difficulty, but to broaden the range of hand behaviors under which the robot must maintain the ZPD interaction.

The iterative procedure is formalized in Algorithm 1. Each iteration has two phases. In the robot phase, the opponent pool P is fixed, and the robot policy is trained with PPO against hand policies sampled from the pool. In the hand phase, a new learned hand policy is initialized from the previous learned hand policy and further trained against the current fixed robot policy. The trained hand policy is then added to P, increasing the diversity of hand behaviors used in the next robot training phase.

```text
Algorithm 1: Iterative League Training
Initialize: opponent pool P <- {pi_H^SPC}, robot policy pi_R^(0)
for n = 1, 2, ..., N do
    Robot phase: train pi_R^(n) with PPO against opponents sampled from P
    Hand phase: warm-start pi_H^(n) from pi_H^(n-1)
    Fine-tune pi_H^(n) against frozen pi_R^(n)
    P <- P union {pi_H^(n)}
end for
```

Uniform sampling from the learned-hand pool is inefficient because different learned hand policies provide different levels of challenge to the current robot. The implemented PFSP rule uses recent episode length as the competitiveness signal. For learned hand i, let l_bar_i denote its mean episode length in a rolling window. If fewer than 20 recent episodes are available for a learned hand, its length estimate is replaced by the median of the available estimates. If no estimates are available, the learned-hand distribution is uniform. The priority score and learned-hand sampling distribution are defined in (8).

l_ref = max_j l_bar_j,   gamma = alpha_PFSP / tau,
s_i = (l_ref / max(l_bar_i, 1))^gamma,
r_i = s_i / sum_j s_j,
p_tilde_i = (1 - mu) r_i + mu / m. (8)

Here, m is the number of learned hand policies, alpha_PFSP controls the strength of prioritization, tau is the temperature, and mu is the uniform exploration mass. A shorter recent episode length yields a larger priority score, so learned hand policies that still challenge the current robot are sampled more often. The uniform mixture keeps every learned hand policy available.

The SPC is sampled outside the learned-hand PFSP distribution with a fixed probability p_SPC. The final opponent-sampling rule is given in (9).

Pr(SPC) = p_SPC,   Pr(H_i) = (1 - p_SPC) p_tilde_i. (9)

In our implementation, the rolling window contains 2000 episodes, alpha_PFSP = 1.0, tau = 1.0, mu = 0.05, and p_SPC = 0.20. Keeping the SPC in the pool reduces the risk that the robot overfits to artifacts of learned hand policies, while PFSP concentrates learned-policy sampling on opponents that remain challenging for the current robot generation.

## IV. EXPERIMENTS AND RESULTS

The evaluation combines simulation experiments with a physical deployment test. The simulation experiments measure robustness across hand behaviors and isolate the contribution of temporal patient-history encoding. The physical deployment test evaluates whether the learned policy can run inside the real-time perception-control stack on the UR10 platform.

### A. Simulation Protocol and Metrics

All simulation experiments use the pursuit-evasion rehabilitation environment described in Section II. The robot controls the magnetic microrobot, and the simulated hand is controlled by the SPC, a learned hand policy, or the mouse-controlled hand interface. The therapeutic objective is defined by a ZPD distance band of 3.5-5.5 workspace units. A step is counted as in-zone when the hand-microrobot distance lies inside this band.

The primary evaluation metric is Time-In-Zone (TIS), defined as the fraction of the episode horizon during which the interaction remains inside the ZPD band. We also report ZPD coverage, episode length, catch rate, and empirical robustness across the learned hand pool. These metrics capture complementary aspects of adaptive difficulty regulation. TIS measures sustained therapeutic interaction over the full horizon. ZPD coverage measures the quality of the realized portion of an episode, while episode length captures survival. Catch rate identifies overly easy interactions in which the microrobot is caught too quickly.

The protocol comparison isolates the effect of opponent diversity. The SPC-only baseline is trained only against the SPC, the single-hand baseline against one learned hand policy, and the league policy through iterative opponent-pool expansion. The representation ablation compares an MLP policy, a GRU policy with temporal interaction history, and a GRU+Aux policy with the auxiliary future-dynamics head. Physical deployment uses the same observation definition but is reported separately in Section IV.D because it introduces camera latency, robot communication, and real-time control constraints that are absent from the simulation-only protocol.

### B. League Training and Cross-Hand Robustness

The league evaluation tests whether opponent-pool training reduces the brittle specialization that can occur when the robot is optimized against a single hand model. The robot is trained for ten generations while the hand pool is expanded with learned hand policies.

League training improves both average behavior and difficult-case behavior, as shown in Fig. 1. The cross-iteration validation matrix evaluates each robot generation against each learned hand policy. Earlier learned hand policies remain comparatively easy to regulate, whereas later policies introduce pursuit strategies that are harder for narrow training regimes. Later robot generations improve across a broader portion of this matrix, indicating that the league does not merely produce a policy that performs well on the final learned hand policy.

Lower-tail robustness is particularly important for rehabilitation. Mean TIS improves across generations, but worst-hand TIS and CVaR20 also improve. A controller that raises only the average score could still fail for patients with more difficult movement patterns. The frontier view shows that later robot generations move toward policies that retain useful ZPD regulation even on harder hand policies.

The final-generation results clarify the remaining failure modes. Some hand policies still create episodes that become too close, while others push the robot toward too-far interactions. Separating these modes matters because both reduce therapeutic value for different reasons. Too-close interactions make the task too easy, while too-far interactions make the task discouraging or unreachable. The league policy maintains useful ZPD coverage despite these different failure patterns, suggesting that the learned behavior is balanced rather than tuned to one type of opponent.

The PFSP sampling traces link this balanced performance to the curriculum induced by opponent sampling. Sampling probability shifts toward learned hand policies that remain challenging for the current robot generation, while the uniform mixture and fixed SPC probability keep earlier and rule-based pursuit behaviors available. As the league evolves, difficult learned opponents receive more attention without removing simpler pursuit behavior from the training distribution.

**Fig. 1. League-training validation and opponent-sampling dynamics for the no-opponent-ID ZPD 3.5-5.5 simulation. (a) Cross-iteration TIS matrix between robot generations and learned hand generations. (b) Mean, worst-hand, and CVaR20 TIS across robot generations. (c) Robustness frontier relating mean TIS, worst-hand TIS, and CVaR20. (d) Final-generation failure decomposition showing too-close rate, too-far rate, and ZPD coverage across test hands. (e) PFSP sampling-probability snapshots during selected training iterations.**

Policy comparison across three hand test sets provides a stricter test of specialization. The final SPC-only, single-hand, and league-trained robot policies are evaluated against the SPC, a learned hand policy, and a mouse-controlled hand. The mouse-controlled condition introduces human reaction time and voluntary pursuit strategy, so it is interpreted as a human-in-the-loop stress test rather than another automated opponent.

**TABLE II: POLICY COMPARISON ACROSS THREE HAND TEST SETS**

| Robot policy | Test set | TIS | ZPD coverage | Episode length | Catch rate |
| --- | --- | --- | --- | --- | --- |
| SPC-only | SPC | 0.31 | 0.54 | 52 | 60% |
| SPC-only | Learned hand | 0.19 | 0.54 | 31 | 90% |
| SPC-only | Mouse hand | 0.14 | 0.50 | 33 | 100% |
| Single-hand | SPC | 0.11 | 0.41 | 25 | 95% |
| Single-hand | Learned hand | 0.49 | 0.60 | 70 | 38% |
| Single-hand | Mouse hand | 0.19 | 0.48 | 40 | 95% |
| League | SPC | 0.38 | 0.55 | 67 | 43% |
| League | Learned hand | 0.48 | 0.65 | 64 | 48% |
| League | Mouse hand | 0.49 | 0.66 | 74 | 60% |

Training against a narrow hand model leads to complementary failure modes across the three test sets. The SPC-only robot remains competitive against the SPC but is caught frequently by the learned and mouse-controlled hands, indicating that rule-based pursuit during training does not cover strategic or human-controlled pursuit behavior. The single-hand robot shows the reverse failure. It performs well against the learned hand policy but degrades sharply against the SPC and mouse-controlled hands.

The league-trained robot is not the best on every scalar metric, but it avoids the severe cross-condition collapse seen in the baselines. Its strongest relative advantage appears under mouse-controlled pursuit, where it produces longer interactions with higher TIS and ZPD coverage while reducing the catch rate from near saturation to 60%. For adaptive rehabilitation, this balanced performance is more important than optimizing a single test condition, because patient behavior can shift across controllers, sessions, and voluntary strategies. The mouse-controlled result remains a stress test rather than a clinical validation, but it provides additional human-in-the-loop evidence that the league policy is more robust to hand behavior outside a single scripted or learned controller.

### C. Network Ablation and Auxiliary Dynamics Analysis

Opponent diversity alone does not determine whether the robot can use the information available in each observation. The ablation study asks whether the robot can regulate difficulty from the current geometric state alone, or whether it needs recent patient-motion history. The MLP baseline observes the current geometric state but does not explicitly encode recent interaction history. The GRU policy processes the 16-frame relative-displacement buffer, allowing it to infer hand velocity, response delay, and pursuit tendency. The GRU+Aux policy further adds the future-dynamics prediction head described in Section III.B.

The main performance gain appears when temporal interaction history is introduced, as shown in Fig. 2. Both recurrent policies learn longer and more rewarding interactions than the MLP baseline, indicating that the robot benefits from observing how the hand has been moving. In this run, the GRU-only and GRU+Aux policies follow similar aggregate learning trends, so temporal sequence modeling is the dominant contributor to control performance.

The auxiliary prediction examples provide a complementary diagnostic of the learned temporal representation. The future-dynamics head captures short-horizon movement direction and diverges gradually at longer horizons, consistent with the uncertainty of multi-step prediction in an interactive task. In this experiment, the auxiliary task encourages the recurrent encoder to represent patient-motion structure and makes that representation inspectable through prediction examples.

**Fig. 2. GRU ablation and auxiliary future-motion prediction. Panels (a) and (b) show smoothed training reward and episode length for MLP, GRU, and GRU+Aux policies trained against the mixed learned-hand and SPC pool. Panels (c) and (d) visualize representative auxiliary future-prediction examples and horizon-dependent trajectory error.**

### D. Real-Time Physical Deployment

The physical deployment test connects the simulation-trained policy to the UR10 platform. Real hand kinematics, camera latency, magnetic actuation, and safety constraints are not fully represented in the virtual training environment. The test therefore focuses on whether the learned policy can run inside a real-time perception-control stack rather than on a paired sim-to-real benchmark.

The physical implementation uses the same 44-dimensional observation structure as simulation. The observation includes robot position from Real-Time Data Exchange (RTDE), hand and microrobot positions from the overhead camera, boundary features, previous action, and the displacement-history buffer. Perception and control run in separate threads to avoid jitter. The camera runs at 10-15 Hz, whereas the robot requires deterministic commands at a fixed control rate. If both processes ran in one thread, a slow You Only Look Once (YOLO) inference step could stall the control loop.

To maintain the 20 Hz control loop, the system uses two threads connected by a single-element message queue. The vision thread captures frames, runs undistortion and detection, and pushes results to the queue. If the queue is full, stale results are replaced. The control thread attempts a non-blocking read each cycle. If a new frame is available, the hand position is updated; otherwise, the system uses dead reckoning based on a low-pass filtered velocity estimate from recent vision updates. The resulting observation is passed to the PPO policy, and the two-dimensional action is mapped through the inverse homography before being sent to the UR10 through the servoL interface.
