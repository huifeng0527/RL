# methodology_ieee_v2.39_formula_table_caption_polish

Stage manuscript draft revised from `methodology_ieee_v2.38_quantified_results_slim_table.md`. This version clarifies CMD-DR formula notation, adds a compact Table II statistics note, removes internal experiment naming from Fig. 3, and reduces numeric detail in result prose.

## II. SYSTEM OVERVIEW AND PROBLEM FORMULATION

We introduce a magnetically actuated, non-contact rehabilitation system for active upper-limb motor training. The user interacts with an unpowered magnetic microrobot on a tabletop workspace, while a robotic arm drives the microrobot indirectly through a permanent magnet placed below the surface. This layout preserves an active pursuit task for the user while avoiding rigid physical coupling between the robot and the human body. It places the user in a pursuit role rather than a passive following role, aligning the task with the need for active engagement in motor rehabilitation. We formulate the control problem as a reinforcement learning (RL) task in which the robot regulates interaction difficulty through the motion of the microrobot.

### A. Hardware Platform

The platform separates actuation, interaction, and perception into three physical layers. In the actuation layer, a UR10 robotic arm is placed beneath the workspace, and a permanent magnet is mounted on its end-effector to drive the microrobot above. In the interaction layer, a 5-mm acrylic sheet supports the unpowered microrobot and forms a physical barrier between the robotic arm and the user. In the perception layer, an overhead red-green-blue (RGB) camera sends images to a host computer, which estimates the positions of the microrobot and the hand and transmits target pose commands to the UR10 over Ethernet.

This layout supports both safety and reproducibility. Fig. 1 summarizes the complete simulation-to-real workflow, from policy training in the pursuit-evasion simulator to camera-based observation construction and physical magnetic actuation. The acrylic sheet creates a fixed separation between the actuator and the user-side workspace, so the participant interacts only with the passive microrobot rather than with the robot arm. The microrobot is lightweight and unpowered, and commanded motion can be limited in software while the UR10 remains below the surface with its standard emergency stop available. The hardware also requires only a collaborative robot, a camera, a calibrated planar workspace, and a permanent magnet. It does not rely on wearable components, force sensors, onboard batteries, or custom actuators, which makes the setup easier to reproduce across sessions.

**Fig. 1. System framework for simulation-to-real non-contact rehabilitation. The simulation environment defines the pursuit-evasion interaction, the policy network maps spatial and temporal observations to microrobot actions, the image-to-observation module converts camera measurements into policy inputs, and the physical platform executes the learned policy through magnetic actuation.**

### B. Problem Formulation

We cast the task as a closed-loop pursuit-evasion interaction on a flat workspace. The participant's hand acts as the pursuer, and the microrobot acts as a moving target driven by the robot arm beneath the acrylic surface. The participant attempts to approach and catch the microrobot, while the robot controls the microrobot motion to keep the task challenging but still reachable. This differs from static reaching because the target does not remain fixed after a movement is planned. The participant must continuously track, anticipate, and correct hand motion, making the task a sustained visuomotor exercise rather than an isolated point-to-point reach [1].

To keep the task challenging but achievable, we define a Zone of Proximal Development (ZPD) as a target distance range between the hand and the microrobot [2]. The robot should neither escape completely nor yield passively. Instead, it should keep the interaction inside this range so that the user remains engaged without the task becoming trivial or unreachable. Hand speed, pursuit strategy, response delay, and motion noise can all change during interaction, making fixed difficulty settings inadequate.

Conventional controllers are limited in this setting because methods such as artificial potential fields or rule-based impedance control usually rely on hand-crafted responses to instantaneous state feedback. These methods struggle to anticipate strategic pursuit behaviors such as interception or to adapt to heterogeneous movement patterns. Their performance can also degrade under delayed response or noisy motion, since purposeful movement and perturbation are not explicitly separated. These limitations motivate an RL formulation in which the controller optimizes a long-horizon ZPD objective under variable hand behavior rather than reacting only to the current hand-microrobot distance.

For robot training, the reward is written as the sum of four intuitive terms: distance regulation, action smoothness, workspace-boundary safety, and catch avoidance. Let d_t be the hand-microrobot distance, a_t be the robot action, and p_R,t be the microrobot position. The robot reward optimized by PPO is summarized as

$$r_t = r_{\mathrm{dist}}(d_t) + r_{\mathrm{smooth}}(a_t) + r_{\mathrm{bound}}(p_{R,t}) + r_{\mathrm{catch}}(d_t). \tag{1}$$

In (1), r_dist(d_t) is the distance-regulation term, which rewards separations inside the ZPD band and penalizes distances that are too close or too far. The term r_smooth(a_t) is the action-smoothness term, which discourages abrupt robot motion. The term r_bound(p_R,t) is the workspace-boundary term, which penalizes the microrobot for leaving the valid workspace. The term r_catch(d_t) is the catch-avoidance term, which penalizes episodes in which the hand reaches the microrobot. Episodes terminate when the microrobot is caught, leaves the valid workspace, or reaches the maximum horizon.

## III. METHODOLOGY

The robot must therefore regulate difficulty while hand behavior changes in both strategy and motor execution. Our approach treats this variability as a simulation problem, an observation problem, and a training-distribution problem. Cognitive-Motor Decoupled Domain Randomization (CMD-DR) broadens the simulated user distribution by separating movement intent from motor-execution constraints. A dual-stream encoder makes hand-motion history available to the policy, while an auxiliary future-dynamics head adds a direct learning signal for short-horizon motion prediction. Iterative league training exposes the robot to a growing pool of hand policies so that the learned controller does not depend on one fixed virtual user.

### A. Cognitive-Motor Decoupled Domain Randomization

Users can differ both in how they choose to move and in how accurately they execute those movements, making it difficult for a single model to capture all variations. CMD-DR separates these two sources of variability by generating intended hand motion first and then applying motor-execution constraints, observation noise, and temporal delay. At the strategy level, a hand policy sampled from the hand-policy pool determines the intended displacement at each timestep. The pool contains a stochastic scripted pursuit controller (SPC) and trained RL hand policies. The SPC provides a transparent pursuit model, while learned hand policies can produce more anticipatory strategies such as interception.

The SPC samples a stride rho_ep once per episode and uses it for the whole episode. At each timestep, it chooses a unit direction q_t and moves one stride along that direction:

$$u_t^{\mathrm{SPC}}=\rho_{\mathrm{ep}}q_t. \tag{2}$$

With probability 1-epsilon, q_t points from the current hand position toward the microrobot. With probability epsilon, q_t is a random unit direction. In our experiments, rho_ep is sampled uniformly between 0.45 and 0.70, and epsilon = 0.05. The learned hand policies replace (2) with their policy outputs, but the resulting intended displacement is passed through the same motor-execution model.

The raw desired displacement is first passed through a first-order low-pass filter to model motion inertia, as shown in (3).

$$x_t = \alpha x_{t-1} + (1-\alpha)u_t. \tag{3}$$

where u_t denotes the raw desired displacement, x_t denotes the filtered displacement, and alpha in (0,1) is the inertia coefficient. Acceleration clipping is then applied componentwise to limit abrupt changes in motion, as shown in (4).

$$\Delta x_t = \mathrm{clip}\left(x_t-x_{t-1}, -a_{\max}, a_{\max}\right). \tag{4}$$

The delayed displacement is then applied to update the simulated hand position,

$$p_{H,t+1}=p_{H,t}+\Delta x_{t-\delta}+\eta_t. \tag{5}$$

where delta is sampled from the delay range and eta_t is zero-mean Gaussian position noise with covariance sigma^2 I. The delay buffer advances at 8 Hz in the motor-execution layer, so the 0--3 frame range in Table I corresponds to 0--375 ms. Table I lists the values and sampled ranges used for these constraints. Through this decoupled design, the robot policy is exposed to plausible hand-motion variations without tying training to a single simulated hand model.

**TABLE I: CMD-DR MOTOR RANDOMIZATION PARAMETERS**

| Parameter | Symbol | Range |
| --- | --- | --- |
| Motion inertia | alpha | 0.7 |
| Maximum acceleration | a_max | 0.15 |
| Gaussian observation noise | sigma | Uniform distribution over 0.01-0.08 |
| Neural delay | delta | 0-3 frames (0-375 ms) |

Uniform distribution denotes sampling uniformly over the stated interval.

### B. Dual-Stream Encoder with Future Dynamics Head

The proposed domain randomization scheme exposes the robot to diverse interaction behaviors during training. However, the robot does not observe the underlying behavior model directly. It must infer behavior patterns from the current interaction state and recent hand motion. The current hand position alone cannot distinguish deliberate pursuit from sensing noise, motor delay, or slow response. We therefore encode the observation with two streams, one for instantaneous geometric context and one for recent hand-motion history.

At each timestep, the 44-dimensional observation vector is divided into a 12-dimensional scalar vector and a 32-dimensional temporal buffer, as shown in (6).

$$o_t = \left[s_t ; h_{t-T:t}\right]. \tag{6}$$

The scalar component represents the instantaneous geometry of the interaction and is defined in (7).

$$s_t = \left[p_R; p_H; d(R,H); b_N,b_S,b_E,b_W; \mathrm{stride}; a_{t-1}\right]. \tag{7}$$

Here, p_R and p_H denote the two-dimensional positions of the microrobot and the hand. d(R,H) is their Euclidean distance. b_N, b_S, b_E, and b_W denote signed distances to the workspace boundaries. The variables stride and a_{t-1} represent the current movement step size and the previous robot action, respectively.

The temporal buffer h_{t-T:t} contains the T = 16 most recent relative hand displacement vectors, forming a 32-dimensional sequence. This history allows the encoder to estimate motion direction, response delay, and oscillatory behavior that are unavailable from the current position alone. Relative displacements make the history representation translation-invariant, so the recurrent stream focuses on how the hand is moving rather than where the interaction occurs in the workspace.

The scalar vector and temporal buffer are encoded in separate streams, as shown in Fig. 2. A multi-layer perceptron (MLP) extracts geometric information from the scalar state, while a gated recurrent unit (GRU) processes the displacement sequence. The two stream outputs are concatenated and passed through a fusion MLP to produce the shared representation used by the policy and value heads.

**Fig. 2. Dual-stream policy architecture with auxiliary future dynamics. The spatial branch encodes the geometric state, the temporal branch encodes recent interaction history with a recurrent module, and the fused representation supports both the policy-value heads and the auxiliary future-motion prediction head.**

The ZPD-based reward provides the main control signal, but it only supervises the encoder indirectly through policy optimization. To provide a denser learning signal for motion representation, we attach a future-dynamics head to the fused representation. The head predicts the relative hand displacement over an eight-step horizon. Because future displacements are available from the rollout, this auxiliary task is self-supervised and requires no additional annotation.

The trajectory prediction loss is given in (8).

$$L_{\mathrm{traj}} = \mathbb{E}\left[\left\|\hat{D}_{t+1:t+H}-D_{t+1:t+H}\right\|_2^2\right], \quad H=8. \tag{8}$$

where D_hat_{t+1:t+H} is the predicted future displacement sequence and D_{t+1:t+H} is the observed future displacement sequence. The full training objective combines the PPO loss with the auxiliary trajectory-prediction loss in (9).

$$L_{\mathrm{total}} = L_{\mathrm{PPO}} + \lambda_{\mathrm{traj}}L_{\mathrm{traj}}. \tag{9}$$

where lambda_traj controls the auxiliary loss weight. Because the future-dynamics head shares the encoder with the policy, its gradients also update the spatial and temporal streams. These auxiliary gradients encourage the encoder to capture motion inertia, delay, and interaction dynamics, helping the policy anticipate near-future hand motion from recent history.

### C. League Training

Even with a well-designed motion representation, achieving stable and adaptive difficulty regulation remains challenging when the training distribution lacks diversity in hand behaviors. A robot trained against a single hand policy may over-specialize to that policy and fail when the pursuit strategy changes. At the same time, training the robot and hand policies simultaneously creates a non-stationary learning problem. The actions of both agents jointly determine the next environment state, so each agent’s transition dynamics and resulting returns are directly influenced by the current policy of the other agent. When both policies are updated together, each update changes the environment faced by the other policy, violating the stationarity assumption of PPO. In practice, the interaction can enter a cycle in which the robot exploits the current hand policy, the hand adapts, and the robot loses strategies that were useful against earlier hands.

We address these two issues separately. To reduce non-stationarity, one side of the interaction is frozen while the other is trained. During robot training, the hand policies are fixed and treated as part of the environment. During hand training, the robot policy is fixed. To reduce over-specialization, the robot is trained against a hand-policy pool P that retains the SPC and learned hand policies from different training stages. Each hand policy is a scripted or learned controller that generates hand motion in simulation. A pool with the SPC and multiple learned hand policies exposes the robot to direct pursuit behavior as well as more strategic interception patterns.

The training objective also differs from that of a purely competitive game. The robot should not simply learn to defeat the hand. It should maintain an appropriate level of challenge for users with different movement patterns. For this reason, the hand-policy pool is used not to maximize adversarial difficulty, but to broaden the range of hand behaviors under which the robot must maintain the ZPD interaction.

The iterative procedure is formalized in Algorithm 1. Each iteration has two phases. In the robot phase, the hand-policy pool P is fixed, and the robot policy is trained with PPO against hand policies sampled from the pool. In the hand phase, a new learned hand policy is initialized from the previous learned hand policy and further trained against the current fixed robot policy. The trained hand policy is then added to P, increasing the diversity of hand behaviors used in the next robot training phase.

**Algorithm 1: Iterative League Training**

**Input:** initial robot policy $\pi_R^{(0)}$, scripted pursuit controller $\pi_H^{\mathrm{SPC}}$, number of league iterations $N$.

**Initialize:** hand-policy pool $\mathcal{P} \leftarrow \{\pi_H^{\mathrm{SPC}}\}$.

1. **for** $n=1,2,\ldots,N$ **do**
2. \quad Train robot policy $\pi_R^{(n)}$ with PPO against hand policies sampled from $\mathcal{P}$.
3. \quad Initialize hand policy $\pi_H^{(n)}$ from $\pi_H^{(n-1)}$ when available.
4. \quad Train $\pi_H^{(n)}$ against the frozen robot policy $\pi_R^{(n)}$.
5. \quad Update hand-policy pool $\mathcal{P} \leftarrow \mathcal{P} \cup \{\pi_H^{(n)}\}$.
6. **end for**

Uniform sampling from the learned-hand pool is inefficient because different learned hand policies provide different levels of challenge to the current robot. The implemented Prioritized Fictitious Self-Play (PFSP) rule uses recent episode length as the competitiveness signal. A shorter episode usually means that the hand quickly catches the microrobot or forces early termination, indicating that the current robot policy has difficulty handling that pursuit pattern. This episode-length priority acts as a dynamic curriculum, directing robot training toward the hand policies that are currently most challenging. For learned hand i, let l_bar_i denote its mean episode length in a rolling window. If fewer than 20 recent episodes are available for a learned hand, its length estimate is replaced by the median of the available estimates. If no estimates are available, the learned-hand distribution is uniform. The priority score and learned-hand sampling distribution are defined in (10).

$$\bar{\ell}_{\mathrm{ref}} = \max_j \bar{\ell}_j, \quad \gamma = \frac{\alpha_{\mathrm{PFSP}}}{\tau},$$
$$s_i = \left(\frac{\bar{\ell}_{\mathrm{ref}}}{\max(\bar{\ell}_i,1)}\right)^\gamma, \quad r_i = \frac{s_i}{\sum_j s_j},$$
$$\tilde{p}_i = (1-\mu)r_i + \frac{\mu}{m}. \tag{10}$$

Here, m is the number of learned hand policies, alpha_PFSP controls the strength of prioritization, tau is the temperature, and mu is the uniform exploration mass. The PFSP coefficient alpha_PFSP is a sampling-priority exponent and is unrelated to the motor-inertia coefficient alpha in Table I. A shorter recent episode length yields a larger priority score, so learned hand policies that still challenge the current robot are sampled more often. The uniform mixture keeps every learned hand policy available.

The SPC is sampled outside the learned-hand PFSP distribution with a fixed probability p_SPC. The final hand-policy sampling rule is given in (11).

$$\Pr(\mathrm{SPC}) = p_{\mathrm{SPC}}, \quad \Pr(H_i) = (1-p_{\mathrm{SPC}})\tilde{p}_i. \tag{11}$$

In implementation, the PFSP estimate uses a finite rolling window, a small uniform exploration mass, and a fixed SPC sampling probability. Keeping the SPC in the pool reduces the risk that the robot overfits to artifacts of learned hand policies, while PFSP concentrates learned-policy sampling on hand policies that remain challenging for the current robot generation.

## IV. EXPERIMENTS AND RESULTS

The evaluation combines simulation experiments with a physical deployment test. The simulation experiments measure robustness across hand behaviors and isolate the contribution of temporal hand-history encoding. The physical deployment test evaluates whether the learned policy can run inside the real-time perception-control stack on the UR10 platform and whether the realized motion responds coherently to measured hand movement.

### A. Simulation Protocol and Metrics

Simulation experiments were conducted in the pursuit-evasion environment described in Section II and instantiated with the hand-behavior models introduced in Section III. The robot controlled the microrobot, while the hand trajectory was generated by one of three evaluation interfaces. The first was the stochastic scripted pursuit controller, the second was a trained hand policy, and the third was a manual mouse pursuit interface used for human-in-the-loop stress testing. The ZPD interval was fixed to 3.5-5.5 workspace units, and an interaction step was classified as in-zone when the Euclidean distance between the hand and the microrobot fell within this interval. The simulator and physical deployment use the same 15 x 10 planar coordinate frame, so one workspace unit corresponds to 1 cm in the calibrated physical workspace. Thus, the 3.5-5.5 workspace-unit ZPD maps to 3.5-5.5 cm in the physical rollout analysis.

Time-In-Zone (TIZ) was used as the primary outcome measure and was computed as the fraction of the full episode horizon spent inside the ZPD interval. Additional metrics were included to separate distinct failure modes of adaptive difficulty regulation. ZPD coverage measures the in-zone fraction over the realized portion of an episode, episode length measures how long the interaction remains active, and catch rate is the fraction of evaluation episodes terminated by catch before the horizon. For league evaluation, robustness across learned hand policies was summarized using mean TIZ, worst-hand TIZ, and Conditional Value-at-Risk over the worst 20% of hand policies (CVaR20) of TIZ.

Two simulation comparisons were performed. The training-protocol comparison evaluated a robot trained only with the stochastic scripted pursuit controller, a robot trained against a single learned hand policy, and a robot trained through iterative league expansion. The representation comparison evaluated a feed-forward MLP policy, a GRU policy that encodes recent hand displacement, and a GRU policy trained with the auxiliary future-dynamics objective. The physical UR10 experiment used the same observation definition but was analyzed separately because camera latency, robot communication, and fixed-rate control introduce constraints that are absent from the simulation protocol.

### B. League Training and Cross-Hand Robustness

The league evaluation tests whether hand-policy pool training reduces the brittle specialization that can occur when the robot is optimized against a single hand model. The robot is trained for ten generations while the hand pool is expanded with learned hand policies.

League training improves both average and difficult-case behavior in the cross-hand evaluation. Mean, worst-hand, and lower-tail TIZ all increase from early to final robot generations, suggesting reduced specialization to a single pursuit style. The final-generation failure decomposition further indicates that the league policy balances too-close and too-far errors while preserving useful ZPD coverage.

The PFSP sampling traces explain how this robustness emerges. Sampling probability shifts toward hand policies that remain challenging for the current robot generation, while the uniform mixture and fixed SPC probability keep earlier and rule-based pursuit behaviors in the training distribution.

**Fig. 3. League-training validation and opponent-sampling dynamics under the ZPD 3.5-5.5 simulation setting. (a) Cross-iteration TIZ matrix between robot generations and learned hand generations. (b) Mean, worst-hand, and CVaR20 TIZ across robot generations. (c) Robustness frontier relating mean TIZ, worst-hand TIZ, and CVaR20. (d) Final-generation failure decomposition showing too-close rate, too-far rate, and ZPD coverage across test hands. (e) PFSP sampling-probability snapshots during selected training iterations.**

Policy comparison across three hand test sets provides a stricter test of specialization. The final SPC-only, single-hand, and league-trained robot policies are evaluated against the SPC, a learned hand policy, and manual mouse pursuit, as summarized in Table II. Manual mouse pursuit introduces human reaction time and voluntary pursuit strategy, so it is interpreted as a human-in-the-loop stress test rather than another automated hand policy. Higher TIZ and episode length are preferred.

**TABLE II: POLICY COMPARISON ACROSS THREE HAND TEST SETS**

| Robot policy | Test set | TIZ | Length |
| --- | --- | ---: | ---: |
| SPC-only | SPC | 0.31 | 52 |
| SPC-only | Learned hand | 0.19 | 31 |
| SPC-only | Manual mouse | 0.14 | 33 |
| Single-hand | SPC | 0.11 | 25 |
| Single-hand | Learned hand | 0.49 | 70 |
| Single-hand | Manual mouse | 0.19 | 40 |
| League | SPC | 0.38 | 67 |
| League | Learned hand | 0.48 | 64 |
| League | Manual mouse | 0.49 | 74 |

Each value is reported as a mean over 50 evaluation episodes for each test condition. TIZ denotes the mean fraction of the full episode horizon spent in the ZPD, and Length denotes mean episode length in simulation steps. Standard deviations are omitted to keep the policy comparison compact.

Training against a narrow hand model leads to complementary failure modes across the three test sets. The SPC-only robot remains competitive on the scripted pursuit controller but produces short, low-TIZ interactions against the learned-hand and manual-mouse test sets. The single-hand robot follows the opposite pattern. It performs well against the learned hand policy but degrades sharply against the SPC and manual mouse pursuit, suggesting specialization to one pursuit style.

The league-trained robot avoids this cross-condition collapse. It achieves the highest TIZ and longest episodes on the SPC and manual mouse pursuit tests, while remaining comparable to the single-hand robot on the learned-hand test. The manual mouse pursuit result is especially informative because it introduces human reaction time and voluntary pursuit strategy. For adaptive difficulty regulation, this balanced profile is more important than optimizing a single test condition, because user behavior can shift across controllers, sessions, and voluntary strategies.

### C. Network Ablation and Auxiliary Dynamics Analysis

Opponent diversity alone does not determine whether the robot can use the information available in each observation. The ablation study asks whether the robot can regulate difficulty from the current geometric state alone, or whether it needs recent hand-motion history. The MLP baseline observes the current geometric state but does not explicitly encode recent interaction history. The GRU policy processes the 16-frame relative-displacement buffer, allowing it to infer hand velocity, response delay, and pursuit tendency. The auxiliary GRU policy further adds the future-dynamics prediction head described in Section III.B.

Temporal interaction history improves the learned controller beyond the MLP baseline. The GRU policy improves task quality relative to the feed-forward encoder, and the auxiliary GRU gives the strongest TIZ and workspace-coverage performance. The auxiliary prediction examples provide a diagnostic of the learned temporal representation. By predicting short-horizon hand displacement from the recurrent state, the auxiliary head trains the encoder through a self-supervised motion-prediction signal in addition to policy gradients. The auxiliary GRU also produces smoother control in the final evaluation.

**Fig. 4. GRU ablation and auxiliary future-motion prediction. Panels (a) and (b) show smoothed training reward and episode length for the MLP policy, the GRU policy, and the auxiliary GRU policy trained against the mixed learned-hand and SPC pool. Panel (c) visualizes representative auxiliary future-prediction examples. Panel (d) summarizes final task-quality metrics normalized to the MLP baseline.**

### D. Real-Time Physical Deployment

The simulation-trained policy was deployed on the UR10 microrobot platform to evaluate zero-shot physical execution under real hand input. A healthy pilot pursued the microrobot on the acrylic workspace while the UR10 drove the magnetic actuator below the surface. The policy was transferred without physical fine-tuning. The physical data comprise nine zero-shot rollouts from the same pilot session. Across these rollouts, the camera and UR10 command loops operated near their target real-time rates, PPO inference required only a few milliseconds, and the image-to-control age remained within the range needed for online interaction.

The implementation preserves the simulation observation layout without adding UR10 telemetry or end-effector pose to the learned-policy input. At each control step, the observation was constructed from calibrated camera measurements of the hand and microrobot, workspace-boundary features, the previous executed robot command, and recent relative hand-displacement history, matching the categories in (6) and (7). The deployment controller used RTDE pose readback only to execute and monitor Cartesian target commands over Ethernet, not as an additional policy observation.

Tracking and calibration were handled by the camera-based perception stack. The microrobot was detected with the trained YOLO model, the hand was detected with MediaPipe-based hand tracking, and a homography calibration mapped image coordinates into the 15 x 10 cm workspace. When a new hand detection was not available, the controller used low-pass dead-reckoning from the last measured hand velocity until the next camera update. Safety constraints included workspace clipping of commanded targets, a maximum commanded target step of 0.6 cm, automatic stop-on-catch at a 1.5 cm distance threshold, UR10 servoStop on shutdown or interruption, and the standard hardware emergency stop available during testing.

Fig. 5 presents one representative zero-shot physical rollout among the nine physical trials. The image sequence in Fig. 5(a) shows the hand and microrobot at selected moments of the same trial. These frames are not independent snapshots; they are time-aligned with the dynamic traces in Fig. 5(b). The selected events cover the beginning of the interaction, a steady pursuit phase, a faster hand movement phase, a slower movement phase, and the end of the rollout. Together, the frames provide direct visual evidence that the policy was executed on the physical platform and that the hand, microrobot, camera perception, and UR10 actuation were coupled in a closed loop.

The quantitative traces in Fig. 5(b) summarize how the interaction evolved during the same physical rollout. The smoothed hand-microrobot distance remains near the target ZPD band for a substantial portion of the trial and repeatedly returns toward this range after deviations. This pattern matches the intended difficulty-regulation objective: the microrobot does not simply move away from the hand indefinitely, nor does it remain static until being caught. Instead, the physical interaction alternates between periods in which the hand closes the distance and periods in which the robot response increases the separation.

The speed profiles provide additional evidence that the measured distance changes are tied to real pursuit dynamics. The hand-speed trace identifies periods of steady motion, acceleration, and deceleration by the pilot, while the robot-speed trace reports the realized motion of the magnetic actuation system. The relative timing of these curves demonstrates that changes in human pursuit speed are reflected in the physical interaction and in the robot response during the same rollout. Thus, the deployment result verifies more than offline policy inference: it demonstrates that the simulation-trained controller can sustain an online hand-microrobot interaction on the UR10 platform using real camera feedback.

**Fig. 5. Zero-shot physical deployment on the UR10 microrobot platform. (a) Key deployment frames from a representative rollout among nine physical trials. (b) Smoothed hand-microrobot distance, target ZPD band, and measured hand and robot speeds from the same rollout.**
