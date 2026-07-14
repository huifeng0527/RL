# methodology_ieee_v2.29_stage_draft_physical_deployment

Stage manuscript draft revised from `methodology_ieee_v2.28_physical_deployment.md`. This version updates the physical deployment result, restores five-figure numbering, and performs a first abbreviation and cross-reference pass.

## II. SYSTEM OVERVIEW AND PROBLEM FORMULATION

We introduce a magnetically actuated, non-contact rehabilitation system for active upper-limb motor training. The patient interacts with an unpowered magnetic microrobot on a tabletop workspace, while a robotic arm drives the microrobot indirectly through a permanent magnet placed below the surface. This layout preserves an active pursuit task for the patient while avoiding rigid physical coupling between the robot and the human body. We formulate the control problem as a reinforcement learning (RL) task in which the robot regulates interaction difficulty through the motion of the magnetic microrobot.

### A. Hardware Platform

The platform separates actuation, interaction, and perception into three physical layers. In the actuation layer, a UR10 robotic arm is placed beneath the workspace, and a permanent magnet is mounted on its end-effector to drive the microrobot above. In the interaction layer, a 5-mm acrylic sheet supports the unpowered magnetic microrobot and forms a physical barrier between the robotic arm and the patient. In the perception layer, an overhead red-green-blue (RGB) camera sends images to a host computer, which estimates the positions of the microrobot and the hand and transmits target pose commands to the UR10 over Ethernet.

This layout supports both safety and reproducibility. Fig. 1 summarizes the complete simulation-to-real workflow, from policy training in the pursuit-evasion simulator to camera-based observation construction and physical magnetic actuation. The acrylic interface reduces risks associated with direct-contact devices, including impact, excessive traction, and local compression. The hardware also requires only a collaborative robot, a camera, and a permanent magnet. It does not rely on wearable components, force sensors, onboard batteries, or custom actuators. Passive magnetic actuation keeps the patient-side device lightweight and unpowered, which simplifies setup and limits the consequences of controller errors during development.

**Fig. 1. System framework for simulation-to-real non-contact rehabilitation. The simulation environment defines the pursuit-evasion interaction, the policy network maps spatial and temporal observations to microrobot actions, the image-to-observation module converts camera measurements into policy inputs, and the physical platform executes the learned policy through magnetic actuation.**

### B. Problem Formulation

We formulate the rehabilitation task as a pursuit-evasion interaction on a flat workspace. The patient sits in front of the acrylic surface and uses one hand to chase the magnetic microrobot, which is controlled by the robot arm below. The task goal is to bring the hand within a small contact radius of the microrobot. Unlike static reaching exercises, the microrobot actively evades the patient's hand. The task therefore requires sustained visuomotor coordination rather than isolated point-to-point reaching [1].

To keep the task challenging but achievable, we define a Zone of Proximal Development (ZPD) as a target distance range between the hand and the microrobot [2]. The robot should neither escape completely nor yield passively. Instead, it should keep the interaction inside this range so that the patient remains engaged without becoming fatigued or frustrated. Patient speed, pursuit strategy, response delay, and motor noise can all change during interaction, making fixed difficulty settings inadequate.

Conventional controllers are limited in this setting because methods such as artificial potential fields or rule-based impedance control usually rely on hand-crafted responses to instantaneous state feedback. These methods struggle to anticipate strategic pursuit behaviors such as interception or to adapt to heterogeneous movement patterns. Their performance can also degrade under tremor, delayed response, or bradykinesia, since purposeful movement and involuntary perturbation are not explicitly separated. These limitations motivate an RL formulation in which the controller optimizes a long-horizon ZPD objective under variable patient behavior rather than reacting only to the current hand-microrobot distance. In our formulation, the robot learns an evasion policy that is rewarded for keeping the interaction within the therapeutic range and penalized for unsafe or uninformative behavior, such as escaping too far or being caught too easily. The key challenge is to learn this policy without assuming a fixed or fully observable patient response model.

## III. METHODOLOGY

The robot must therefore regulate difficulty while patient behavior changes in both strategy and motor execution. Our approach treats this variability as a simulation problem, an observation problem, and a training-distribution problem. Cognitive-Motor Decoupled Domain Randomization (CMD-DR) broadens the simulated patient distribution by separating movement intent from biomechanical constraints. A dual-stream encoder makes patient motion history available to the policy, while an auxiliary future-dynamics head adds a direct learning signal for short-horizon hand-motion prediction. Iterative league training exposes the robot to a growing pool of hand policies so that the learned controller does not depend on one fixed virtual patient.

### A. Cognitive-Motor Decoupled Domain Randomization

In rehabilitation settings, patients can differ both in how they choose to move and in how accurately they can execute those movements, making it difficult for a single model to capture all variations. CMD-DR separates these two sources of variability by generating intended hand motion first and then applying motor-execution constraints, observation noise, and temporal delay. At the strategy level, a hand policy sampled from the opponent pool determines the intended displacement at each timestep. The pool contains a stochastic scripted pursuit controller (SPC) and trained RL hand policies. The SPC provides a transparent pursuit model, while learned hand policies can produce more anticipatory strategies such as interception.

The SPC samples a stride rho_ep once per episode and uses it for the whole episode. At each timestep, it either moves toward the microrobot or makes a small random exploratory move. Its intended displacement is defined in (1), where the pursuit direction points from the current hand position toward the current microrobot position.

$$\hat{e}_t = \frac{p_{R,t}-p_{H,t}}{\|p_{R,t}-p_{H,t}\|_2}, \quad u_t^{\mathrm{SPC}} = \rho_{\mathrm{ep}}\left[z_t \hat{\xi}_t + (1-z_t)\hat{e}_t\right], \quad z_t \sim \mathrm{Bernoulli}(\epsilon). \tag{1}$$

Here, p_R,t and p_H,t are the microrobot and hand positions, xi_hat_t is a normalized random direction, rho_ep is sampled from U(0.45, 0.70), and epsilon = 0.05. The learned hand policies replace (1) with their policy outputs, but the resulting intended displacement is passed through the same motor-execution model.

The raw desired displacement is first passed through a first-order low-pass filter to model muscle inertia, as shown in (2).

$$x_t = \alpha x_{t-1} + (1-\alpha)u_t, \quad \alpha \in (0,1). \tag{2}$$

where u_t denotes the raw desired displacement and x_t denotes the filtered displacement. Acceleration clipping then limits abrupt changes in motion, as shown in (3).

$$\Delta x_t = \mathrm{clip}\left(x_t-x_{t-1}, -a_{\max}, a_{\max}\right). \tag{3}$$

Gaussian observation noise simulates position-estimation errors from the vision system. A sampled temporal delay models delayed motor response by executing a displacement generated several frames earlier. The delay buffer advances at 8 Hz in the motor-execution layer, so the 0--3 frame range in Table I corresponds to 0--375 ms. Table I lists the values and sampled ranges used for these constraints. Through this decoupled design, the robot policy is exposed to diverse patient-like behaviors without tying training to a single simulated hand model.

**TABLE I: CMD-DR MOTOR RANDOMIZATION PARAMETERS**

| Parameter | Symbol | Range |
| --- | --- | --- |
| Muscle inertia | alpha | 0.7 |
| Maximum acceleration | a_max | 0.15 |
| Gaussian observation noise | sigma | U(0.01, 0.08) |
| Neural delay | delta | 0-3 frames (0-375 ms) |

### B. Dual-Stream Encoder with Future Dynamics Head

The proposed domain randomization scheme exposes the robot to diverse interaction behaviors during training, but the robot does not observe the underlying behavior model directly. It must infer behavior patterns from the current interaction state and recent hand motion. The current hand position alone cannot distinguish deliberate pursuit from sensing noise, motor delay, tremor-like perturbations, or slow response. We therefore encode the observation with two streams, one for instantaneous geometric context and one for recent hand-motion history.

At each timestep, the 44-dimensional observation vector is divided into a 12-dimensional scalar vector and a 32-dimensional temporal buffer, as shown in (4).

$$o_t = \left[s_t ; h_{t-T:t}\right]. \tag{4}$$

The scalar component represents the instantaneous geometry of the interaction and is defined in (5).

$$s_t = \left[p_R; p_H; d(R,H); b_N,b_S,b_E,b_W; \mathrm{stride}; a_{t-1}\right]. \tag{5}$$

Here, p_R and p_H denote the two-dimensional positions of the microrobot and the hand. d(R,H) is their Euclidean distance. b_N, b_S, b_E, and b_W denote signed distances to the workspace boundaries. The variables stride and a_{t-1} represent the current movement step size and the previous robot action, respectively.

The temporal buffer h_{t-T:t} contains the T = 16 most recent relative hand displacement vectors, forming a 32-dimensional sequence. This history allows the encoder to estimate motion direction, response delay, and oscillatory behavior that are unavailable from the current position alone. Relative displacements make the history representation translation-invariant, so the recurrent stream focuses on how the hand is moving rather than where the interaction occurs in the workspace.

The scalar vector and temporal buffer are encoded in separate streams, as shown in Fig. 2. A multi-layer perceptron (MLP) extracts geometric information from the scalar state, while a gated recurrent unit (GRU) processes the displacement sequence. The two stream outputs are concatenated and passed through a fusion MLP to produce the shared representation used by the policy and value heads.

**Fig. 2. Dual-stream policy architecture with auxiliary future dynamics. The spatial branch encodes the geometric state, the temporal branch encodes recent interaction history with a recurrent module, and the fused representation supports both the policy-value heads and the auxiliary future-motion prediction head.**

The ZPD-based reward provides the main control signal, but it only supervises the encoder indirectly through policy optimization. To provide a denser learning signal for motion representation, we attach a future-dynamics head to the fused representation. The head predicts the relative hand displacement over an eight-step horizon. Because future displacements are available from the rollout, this auxiliary target requires no additional annotation.

The trajectory prediction loss is given in (6).

$$L_{\mathrm{traj}} = \mathbb{E}\left[\left\|\hat{D}_{t+1:t+H}-D_{t+1:t+H}\right\|_2^2\right], \quad H=8. \tag{6}$$

where D_hat_{t+1:t+H} is the predicted future displacement sequence and D_{t+1:t+H} is the observed future displacement sequence. The full training objective combines the Proximal Policy Optimization (PPO) loss with the auxiliary trajectory-prediction loss in (7).

$$L_{\mathrm{total}} = L_{\mathrm{PPO}} + \lambda_{\mathrm{traj}}L_{\mathrm{traj}}. \tag{7}$$

where lambda_traj controls the auxiliary loss weight. Because the future-dynamics head shares the encoder with the policy, its gradients also update the spatial and temporal streams. These auxiliary gradients encourage the encoder to capture motion inertia, delay, and interaction dynamics, helping the policy anticipate near-future hand motion from recent history.

### C. League Training

Even with a well-designed motion representation, achieving stable and adaptive difficulty regulation remains challenging when the training distribution lacks diversity in hand behaviors. A robot trained against a single hand policy may over-specialize to that policy and fail when the pursuit strategy changes. At the same time, training the robot and hand policies simultaneously creates a non-stationary learning problem. The actions of both agents jointly determine the next environment state, so each agent’s transition dynamics and resulting returns are directly influenced by the current policy of the other agent. When both policies are updated together, each update changes the environment faced by the other policy, violating the stationarity assumption of PPO. In practice, the interaction can enter a cycle in which the robot exploits the current hand policy, the hand adapts, and the robot loses strategies that were useful against earlier hands.

We address these two issues separately. To reduce non-stationarity, one side of the interaction is frozen while the other is trained. During robot training, the hand policies are fixed and treated as part of the environment. During hand training, the robot policy is fixed. To reduce over-specialization, the robot is trained against an opponent pool P that retains the SPC and learned hand policies from different training stages. Each hand policy is a scripted or learned controller that generates hand motion in simulation. A pool with the SPC and multiple learned hand policies exposes the robot to direct pursuit behavior as well as more strategic interception patterns.

The rehabilitation objective also differs from that of a purely competitive game. The robot should not simply learn to defeat the hand. It should maintain an appropriate level of challenge for patients with different movement abilities. For this reason, the opponent pool is used not to maximize adversarial difficulty, but to broaden the range of hand behaviors under which the robot must maintain the ZPD interaction.

The iterative procedure is formalized in Algorithm 1. Each iteration has two phases. In the robot phase, the opponent pool P is fixed, and the robot policy is trained with PPO against hand policies sampled from the pool. In the hand phase, a new learned hand policy is initialized from the previous learned hand policy and further trained against the current fixed robot policy. The trained hand policy is then added to P, increasing the diversity of hand behaviors used in the next robot training phase.

**Algorithm 1: Iterative League Training**

**Input:** initial robot policy $\pi_R^{(0)}$, scripted pursuit controller $\pi_H^{\mathrm{SPC}}$, number of league iterations $N$.

**Initialize:** opponent pool $\mathcal{P} \leftarrow \{\pi_H^{\mathrm{SPC}}\}$.

1. **for** $n=1,2,\ldots,N$ **do**
2. \quad Train robot policy $\pi_R^{(n)}$ with PPO against opponents sampled from $\mathcal{P}$.
3. \quad Initialize hand policy $\pi_H^{(n)}$ from $\pi_H^{(n-1)}$ when available.
4. \quad Train $\pi_H^{(n)}$ against the frozen robot policy $\pi_R^{(n)}$.
5. \quad Update opponent pool $\mathcal{P} \leftarrow \mathcal{P} \cup \{\pi_H^{(n)}\}$.
6. **end for**

Uniform sampling from the learned-hand pool is inefficient because different learned hand policies provide different levels of challenge to the current robot. The implemented Prioritized Fictitious Self-Play (PFSP) rule uses recent episode length as the competitiveness signal. For learned hand i, let l_bar_i denote its mean episode length in a rolling window. If fewer than 20 recent episodes are available for a learned hand, its length estimate is replaced by the median of the available estimates. If no estimates are available, the learned-hand distribution is uniform. The priority score and learned-hand sampling distribution are defined in (8).

$$\bar{\ell}_{\mathrm{ref}} = \max_j \bar{\ell}_j, \quad \gamma = \frac{\alpha_{\mathrm{PFSP}}}{\tau},$$
$$s_i = \left(\frac{\bar{\ell}_{\mathrm{ref}}}{\max(\bar{\ell}_i,1)}\right)^\gamma, \quad r_i = \frac{s_i}{\sum_j s_j},$$
$$\tilde{p}_i = (1-\mu)r_i + \frac{\mu}{m}. \tag{8}$$

Here, m is the number of learned hand policies, alpha_PFSP controls the strength of prioritization, tau is the temperature, and mu is the uniform exploration mass. The PFSP coefficient alpha_PFSP is a sampling-priority exponent and is unrelated to the motor-inertia coefficient alpha in Table I. A shorter recent episode length yields a larger priority score, so learned hand policies that still challenge the current robot are sampled more often. The uniform mixture keeps every learned hand policy available.

The SPC is sampled outside the learned-hand PFSP distribution with a fixed probability p_SPC. The final opponent-sampling rule is given in (9).

$$\Pr(\mathrm{SPC}) = p_{\mathrm{SPC}}, \quad \Pr(H_i) = (1-p_{\mathrm{SPC}})\tilde{p}_i. \tag{9}$$

In our implementation, the rolling window contains 2000 episodes, alpha_PFSP = 1.0, tau = 1.0, mu = 0.05, and p_SPC = 0.20. Keeping the SPC in the pool reduces the risk that the robot overfits to artifacts of learned hand policies, while PFSP concentrates learned-policy sampling on opponents that remain challenging for the current robot generation.

## IV. EXPERIMENTS AND RESULTS

The evaluation combines simulation experiments with a physical deployment test. The simulation experiments measure robustness across hand behaviors and isolate the contribution of temporal patient-history encoding. The physical deployment test evaluates whether the learned policy can run inside the real-time perception-control stack on the UR10 platform and whether the realized motion responds coherently to measured hand movement.

### A. Simulation Protocol and Metrics

Simulation experiments were conducted in the pursuit-evasion environment described in Section II and instantiated with the patient-behavior models introduced in Section III. The robot controlled the magnetic microrobot, while the hand trajectory was generated by one of three evaluation interfaces. The first was the stochastic scripted pursuit controller, the second was a trained hand policy, and the third was a mouse-operated hand interface used for human-in-the-loop stress testing. The ZPD interval was fixed to 3.5-5.5 workspace units, and an interaction step was classified as in-zone when the Euclidean distance between the hand and the microrobot fell within this interval.

Time-In-Zone (TIZ) was used as the primary outcome measure and was computed as the fraction of the full episode horizon spent inside the ZPD interval. Additional metrics were included to separate distinct failure modes of adaptive difficulty regulation. ZPD coverage measures the in-zone fraction over the realized portion of an episode, episode length measures how long the interaction remains active, and catch rate measures the frequency with which the microrobot is reached too quickly. For league evaluation, robustness across learned hand policies was summarized using mean TIZ, worst-hand TIZ, and Conditional Value-at-Risk over the worst 20% of hand policies (CVaR20) of TIZ.

Two simulation comparisons were performed. The training-protocol comparison evaluated a robot trained only with the stochastic scripted pursuit controller, a robot trained against a single learned hand policy, and a robot trained through iterative league expansion. The representation comparison evaluated a feed-forward MLP policy, a GRU policy that encodes recent hand displacement, and a GRU policy trained with the auxiliary future-dynamics objective. The physical UR10 experiment used the same observation definition but was analyzed separately because camera latency, robot communication, and fixed-rate control introduce constraints that are absent from the simulation protocol.

### B. League Training and Cross-Hand Robustness

The league evaluation tests whether opponent-pool training reduces the brittle specialization that can occur when the robot is optimized against a single hand model. The robot is trained for ten generations while the hand pool is expanded with learned hand policies.

League training improves both average behavior and difficult-case behavior, as shown in Fig. 3. The cross-iteration validation matrix evaluates each robot generation against each learned hand policy. Earlier learned hand policies remain comparatively easy to regulate, whereas later policies introduce pursuit strategies that are harder for narrow training regimes. Later robot generations improve across a broader portion of this matrix, indicating that the league does not merely produce a policy that performs well on the final learned hand policy.

Lower-tail robustness is particularly important for rehabilitation. Mean TIZ improves across generations, but worst-hand TIZ and CVaR20 also improve. A controller that raises only the average score could still fail for patients with more difficult movement patterns. The frontier view shows that later robot generations move toward policies that retain useful ZPD regulation even on harder hand policies.

The final-generation results clarify the remaining failure modes. Some hand policies still create episodes that become too close, while others push the robot toward too-far interactions. Separating these modes matters because both reduce therapeutic value for different reasons. Too-close interactions make the task too easy, while too-far interactions make the task discouraging or unreachable. The league policy maintains useful ZPD coverage despite these different failure patterns, suggesting that the learned behavior is balanced rather than tuned to one type of opponent.

The PFSP sampling traces link this balanced performance to the curriculum induced by opponent sampling. Sampling probability shifts toward learned hand policies that remain challenging for the current robot generation, while the uniform mixture and fixed SPC probability keep earlier and rule-based pursuit behaviors available. As the league evolves, difficult learned opponents receive more attention without removing simpler pursuit behavior from the training distribution.

**Fig. 3. League-training validation and opponent-sampling dynamics for the no-opponent-ID ZPD 3.5-5.5 simulation. (a) Cross-iteration TIZ matrix between robot generations and learned hand generations. (b) Mean, worst-hand, and CVaR20 TIZ across robot generations. (c) Robustness frontier relating mean TIZ, worst-hand TIZ, and CVaR20. (d) Final-generation failure decomposition showing too-close rate, too-far rate, and ZPD coverage across test hands. (e) PFSP sampling-probability snapshots during selected training iterations.**

Policy comparison across three hand test sets provides a stricter test of specialization. The final SPC-only, single-hand, and league-trained robot policies are evaluated against the SPC, a learned hand policy, and a mouse-controlled hand, as summarized in Table II. The mouse-controlled condition introduces human reaction time and voluntary pursuit strategy, so it is interpreted as a human-in-the-loop stress test rather than another automated opponent.

**TABLE II: POLICY COMPARISON ACROSS THREE HAND TEST SETS**

| Robot policy | SPC TIZ | SPC length | Learned-hand TIZ | Learned-hand length | Mouse-hand TIZ | Mouse-hand length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SPC-only | 0.31 | 52 | 0.19 | 31 | 0.14 | 33 |
| Single-hand | 0.11 | 25 | 0.49 | 70 | 0.19 | 40 |
| League | 0.38 | 67 | 0.48 | 64 | 0.49 | 74 |

Training against a narrow hand model leads to complementary failure modes across the three test sets. The SPC-only robot remains strongest on the scripted pursuit controller but produces short, low-TIZ interactions against the learned and mouse-controlled hands. The single-hand robot shows the opposite pattern. It performs well against the learned hand policy but degrades sharply against the SPC and mouse-controlled hands, indicating specialization to one pursuit style.

The league-trained robot avoids this cross-condition collapse. It achieves the highest TIZ and longest episodes on the SPC and mouse-controlled tests, while remaining comparable to the single-hand robot on the learned-hand test. The mouse-controlled result is especially informative because it introduces human reaction time and voluntary pursuit strategy. For adaptive rehabilitation, this balanced profile is more important than optimizing a single test condition, because patient behavior can shift across controllers, sessions, and voluntary strategies.

### C. Network Ablation and Auxiliary Dynamics Analysis

Opponent diversity alone does not determine whether the robot can use the information available in each observation. The ablation study asks whether the robot can regulate difficulty from the current geometric state alone, or whether it needs recent patient-motion history. The MLP baseline observes the current geometric state but does not explicitly encode recent interaction history. The GRU policy processes the 16-frame relative-displacement buffer, allowing it to infer hand velocity, response delay, and pursuit tendency. The auxiliary GRU policy further adds the future-dynamics prediction head described in Section III.B.

Fig. 4 shows that temporal interaction history improves the robot policy beyond the MLP baseline, and that auxiliary future-dynamics supervision further improves the final task-quality profile. In the training curves, both recurrent policies achieve longer and more stable interactions than the MLP policy, indicating that recent hand-motion history is useful for regulating difficulty. The auxiliary GRU follows a similar training trend to the GRU policy, but its final evaluation is stronger in the task-quality summary: it achieves higher TIZ, broader workspace coverage, and lower jerk relative to the MLP baseline.

The auxiliary prediction examples provide a complementary diagnostic of why this improvement is plausible. The future-dynamics head learns to predict short-horizon hand displacement from the recurrent representation, so the temporal encoder is trained not only through sparse policy gradients but also through a supervised motion-prediction signal. This additional signal encourages the representation to encode pursuit direction, inertia, and short-term changes in hand motion. The result is not merely an interpretable auxiliary output; it also contributes to the final control behavior summarized in Fig. 4(d).

**Fig. 4. GRU ablation and auxiliary future-motion prediction. Panels (a) and (b) show smoothed training reward and episode length for the MLP policy, the GRU policy, and the auxiliary GRU policy trained against the mixed learned-hand and SPC pool. Panel (c) visualizes representative auxiliary future-prediction examples. Panel (d) summarizes final task-quality metrics normalized to the MLP baseline.**

### D. Real-Time Physical Deployment

The simulation-trained policy was deployed on the UR10 magnetic microrobot platform to evaluate zero-shot physical execution under real hand input. During the deployment trial, a healthy pilot pursued the microrobot on the acrylic workspace, while the UR10 drove the magnetic actuator below the surface. The policy was transferred without physical fine-tuning. At each control step, the observation was constructed from calibrated camera measurements and UR10 state feedback, preserving the same observation categories used in simulation, including hand position, microrobot position, robot state, workspace-boundary features, previous action, and recent displacement history.

Fig. 5 presents a representative zero-shot physical rollout. The image sequence in Fig. 5(a) shows the hand and microrobot at selected moments of the same trial. These frames are not independent snapshots; they are time-aligned with the dynamic traces in Fig. 5(b). The selected events cover the beginning of the interaction, a steady pursuit phase, a faster hand movement phase, a slower movement phase, and the end of the rollout. Together, the frames provide direct visual evidence that the policy was executed on the physical platform and that the hand, microrobot, camera perception, and UR10 actuation were coupled in a closed loop.

The quantitative traces in Fig. 5(b) show how the interaction evolved during the same physical rollout. The smoothed hand-microrobot distance remains near the target ZPD band for a substantial portion of the trial and repeatedly returns toward this range after deviations. This pattern matches the intended difficulty-regulation objective: the microrobot does not simply move away from the hand indefinitely, nor does it remain static until being caught. Instead, the physical interaction alternates between periods in which the hand closes the distance and periods in which the robot response increases the separation.

The speed profiles provide additional evidence that the measured distance changes are tied to real pursuit dynamics. The hand-speed trace identifies periods of steady motion, acceleration, and deceleration by the pilot, while the robot-speed trace reports the realized motion of the magnetic actuation system. The relative timing of these curves shows that changes in human pursuit speed are reflected in the physical interaction and in the robot response during the same rollout. Thus, the deployment result verifies more than offline policy inference: it shows that the simulation-trained controller can sustain an online hand-microrobot interaction on the UR10 platform using real camera feedback.

**Fig. 5. Zero-shot physical deployment on the UR10 magnetic microrobot platform. (a) Key deployment frames from a representative rollout. (b) Smoothed hand-microrobot distance, target ZPD band, and measured hand and robot speeds from the same rollout.**
