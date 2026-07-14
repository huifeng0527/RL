# methodology_ieee_v2.54_target_zpd_band_prose

Stage manuscript draft revised from `methodology_ieee_v2.53_restore_fixed_technical_terms.md`. This version standardizes references to the target ZPD band in the prose without adding the numerical setting to a table.

## II. SYSTEM OVERVIEW AND PROBLEM FORMULATION

We introduce a magnetically actuated, noncontact rehabilitation system for active motor training of the upper limb. The user interacts with an unpowered magnetic microrobot on a tabletop workspace, while a robotic arm drives the microrobot indirectly through a permanent magnet placed below the surface. This layout preserves an active pursuit task while avoiding rigid physical coupling between the robot and the human body. The user acts as the pursuer rather than a passive follower, which supports active engagement in motor rehabilitation. We formulate the control problem as a reinforcement learning (RL) task in which microrobot motion regulates interaction difficulty.

### A. Hardware Platform

The platform separates actuation, interaction, and perception into three physical layers. In the actuation layer, a UR10 robotic arm is placed beneath the workspace, and a permanent magnet is mounted on its end effector to drive the microrobot above. In the interaction layer, an acrylic sheet 5 mm thick supports the unpowered microrobot and forms a physical barrier between the robotic arm and the user. In the perception layer, an overhead RGB camera sends images to a host computer. The computer estimates the microrobot and hand positions and transmits target pose commands to the UR10 over Ethernet.

This layout supports both safety and reproducibility. Fig. 1 summarizes the workflow from simulation training to camera observation, policy inference, and physical magnetic actuation. The acrylic sheet creates a fixed separation between the actuator and the workspace used by the participant. The participant therefore interacts only with the passive microrobot rather than with the robot arm. The microrobot is lightweight and unpowered, and software limits can constrain commanded motion while the UR10 remains below the surface with its standard emergency stop available. The hardware requires only a collaborative robot, a camera, a calibrated planar workspace, and a permanent magnet. It does not rely on wearable components, force sensors, onboard batteries, or custom actuators, which facilitates reproduction across sessions.

**Fig. 1. System framework for noncontact rehabilitation from simulation to physical deployment. The simulation environment defines the pursuit interaction, the policy network maps spatial and temporal observations to microrobot actions, the observation module converts camera measurements into policy inputs, and the physical platform executes the learned policy through magnetic actuation.**

### B. Problem Formulation

We cast the task as a pursuit interaction with continuous feedback on a flat workspace. The participant's hand is the pursuer, and the microrobot is a moving target driven beneath the acrylic surface. The participant attempts to approach and catch the microrobot while the robot adjusts target motion to keep the task challenging but reachable. Unlike static reaching, the target continues to move after the participant initiates a movement. The task requires continuous tracking, anticipation, and correction rather than a single reaching movement [1].

The Zone of Proximal Development (ZPD) is defined operationally as a target range for the distance between the hand and microrobot [2]. In this study, the target ZPD band was established after workspace calibration. It was selected for the desktop pilot platform to leave enough separation to avoid immediate catch while keeping the microrobot visually and physically reachable. This operating band is specific to the platform rather than a universal clinical threshold. Use with patients would require calibration during each session according to movement range, speed, and therapeutic goal. The robot should keep the interaction inside the target ZPD band rather than escaping completely or yielding passively. Hand speed, pursuit strategy, response delay, and motion noise can all change during interaction, making fixed difficulty settings inadequate.

Classical artificial potential field controllers and impedance controllers that provide assistance as needed are useful baselines for obstacle avoidance and rehabilitation [3]–[5]. Their assumptions are less well matched to this interactive pursuit task. A potential field based on distance can drive the microrobot directly away from the hand, but it has no explicit model of where an intercepting hand will arrive next. When the hand cuts across the target path, a purely reactive field may respond only after the interception geometry has changed. The resulting trajectory can oscillate, move toward a workspace boundary, or end in an early catch. Impedance control based on fixed rules faces a complementary issue. It can modulate compliance around a desired trajectory or endpoint error, but its gains and reference behaviors require manual design. Under camera delay, noisy hand estimates, or abrupt shifts from direct pursuit to interception, the controller must distinguish deliberate strategy from motor lag or measurement noise. A fixed rule has limited ability to make that distinction or optimize ZPD regulation over several future steps. These failure modes motivate an RL policy that regulates difficulty over a longer horizon under variable hand behavior.

For robot training, the reward is written as the sum of four intuitive terms: distance regulation, action smoothness, workspace safety, and catch avoidance. Let d_t be the distance between the hand and microrobot, a_t be the robot action, and p_R,t be the microrobot position. The robot reward optimized by PPO is summarized as

$$r_t = r_{\mathrm{dist}}(d_t) + r_{\mathrm{smooth}}(a_t) + r_{\mathrm{bound}}(p_{R,t}) + r_{\mathrm{catch}}(d_t). \tag{1}$$

Equation (1) groups the robot reward into four components. The distance component r_dist(d_t) encourages the separation between the hand and microrobot to remain inside the target ZPD band and penalizes deviations on either side. The smoothness component r_smooth(a_t) discourages abrupt commanded motion. The boundary component r_bound(p_R,t) penalizes microrobot positions outside the valid workspace. The catch component r_catch(d_t) penalizes contact between the hand and microrobot. Episodes terminate after catch, workspace exit, or the maximum horizon.

## III. METHODOLOGY

Difficulty regulation must remain stable while hand behavior changes in both strategy and motor execution. The method handles this variability across simulation, observation, and training distribution. The domain randomization strategy separates movement intent from constraints on motor execution. A dual-stream encoder provides recent hand motion to the policy, and an auxiliary dynamics head supplies a direct learning signal for prediction over a short horizon. Iterative league training exposes the robot to a growing pool of hand policies, reducing dependence on one fixed virtual user.

### A. Decoupled Randomization of Cognitive and Motor Factors

Hand behavior varies in intended pursuit strategy and execution accuracy. The proposed domain randomization strategy separates these factors. A hand policy first generates the intended displacement, and the motor execution layer then applies motion constraints, observation noise, and temporal delay. The strategy policy is sampled from a pool containing the SPC and trained RL hand policies. The SPC provides a transparent pursuit model; learned hand policies can produce more anticipatory interception strategies.

For the stochastic scripted pursuit controller, let ρ_ep denote the stride length sampled for each episode and q_t denote a planar unit direction. The intended displacement is

$$u_t^{\mathrm{SPC}}=\rho_{\mathrm{ep}}q_t. \tag{2}$$

The stride ρ_ep is sampled once at episode reset and then held fixed. The direction q_t follows a pursuit rule with ε exploration. With probability 1 - ε, q_t points from the current hand position toward the microrobot. With probability ε, q_t is sampled uniformly from all planar unit directions. In the experiments, ρ_ep is sampled uniformly between 0.45 and 0.70, and ε is fixed at 0.05. Learned hand policies output the intended displacement directly, and the same motor execution model is applied afterward.

Let u_t denote the intended hand velocity, v_t denote the executed velocity, and ṽ_t denote the filtered target velocity. Motion smoothing is applied as

$$\tilde{v}_t = \alpha u_t + (1-\alpha)v_{t-1}. \tag{3}$$

The change from the previous executed velocity is then limited by its Euclidean norm,

$$v_t = v_{t-1} + \mathrm{clip}_{\mathrm{norm}}\left(\tilde{v}_t-v_{t-1}, a_{\max}\right). \tag{4}$$

Here, clip_norm(z, a_max) returns z when its Euclidean norm does not exceed a_max and otherwise rescales it to have norm a_max. The executed velocity updates the simulated hand position,

$$p_{H,t+1}=p_{H,t}+v_t. \tag{5}$$

Equation (5) defines the core kinematic update. The delay and noise parameters in Table I are applied separately by the motor execution layer. This formulation allows a constant intended velocity to converge to a constant executed velocity rather than causing the hand motion to vanish.

**TABLE I: MOTOR RANDOMIZATION PARAMETERS**

| Parameter | Symbol | Range |
| --- | --- | --- |
| Motion inertia | alpha | 0.7 |
| Maximum acceleration | a_max | 0.15 |
| Gaussian observation noise | sigma | Uniform distribution over 0.01–0.08 |
| Neural delay | delta | 0–3 frames (0–375 ms) |

Uniform distribution denotes sampling uniformly over the stated interval.

### B. Dual-Stream Encoder with Auxiliary Dynamics Prediction

The proposed domain randomization scheme exposes the robot to diverse interaction behaviors during training. The robot does not observe the underlying behavior model directly. It must infer behavior patterns from the current interaction state and recent hand motion. Current hand position alone cannot distinguish deliberate pursuit from sensing noise, motor delay, or slow response. The dual-stream encoder separates instantaneous geometric context from the recent history of hand motion.

At each timestep, the observation vector has 44 dimensions and is divided into a scalar vector with 12 dimensions and a temporal buffer with 32 dimensions, as shown in (6).

$$o_t = \left[s_t ; h_{t-T:t}\right]. \tag{6}$$

The scalar component represents the instantaneous geometry of the interaction and is defined in (7).

$$s_t = \left[p_R; p_H; d(R,H); b_N,b_S,b_E,b_W; \mathrm{stride}; a_{t-1}\right]. \tag{7}$$

Here, p_R and p_H denote the 2D positions of the microrobot and the hand. d(R,H) is their Euclidean distance. b_N, b_S, b_E, and b_W denote signed distances to the workspace boundaries. The variables stride and a_{t-1} represent the current movement step size and the previous robot action, respectively.

The temporal buffer h_{t-T:t} contains the T = 16 most recent relative hand displacement vectors, forming a sequence with 32 dimensions. This sequence helps the encoder estimate motion direction, response delay, and oscillatory behavior that are unavailable from the current position alone. Relative displacements make the representation invariant to translation, so the recurrent stream focuses on how the hand is moving rather than where the interaction occurs in the workspace.

The scalar vector and temporal buffer are encoded in separate streams, as shown in Fig. 2. A multilayer perceptron (MLP) extracts geometric information from the scalar state, while a gated recurrent unit (GRU) processes the displacement sequence. The two stream outputs are concatenated and passed through a fusion MLP to produce the shared representation used by the policy and value heads.

**Fig. 2. Dual-stream policy architecture with auxiliary dynamics prediction. The spatial branch encodes the geometric state, the temporal branch encodes recent interaction history with a recurrent module, and the fused representation supports the policy head, value head, and auxiliary prediction head.**

The ZPD reward provides the main control signal, but it supervises the encoder only indirectly through policy optimization. An auxiliary dynamics head adds a denser learning signal for motion representation. The head predicts relative hand displacement over a horizon of eight steps. Future displacements are available from the rollout, so this self-supervised auxiliary task requires no additional annotation.

The trajectory prediction loss is given in (8).

$$L_{\mathrm{traj}} = \mathbb{E}\left[\left\|\hat{D}_{t+1:t+H}-D_{t+1:t+H}\right\|_2^2\right], \quad H=8. \tag{8}$$

Here, D_hat_{t+1:t+H} is the predicted future displacement sequence, and D_{t+1:t+H} is the observed future displacement sequence. The full training objective combines the PPO loss with the auxiliary trajectory prediction loss in (9).

$$L_{\mathrm{total}} = L_{\mathrm{PPO}} + \lambda_{\mathrm{traj}}L_{\mathrm{traj}}. \tag{9}$$

Here, lambda_traj controls the auxiliary loss weight. Because the auxiliary dynamics head shares the encoder with the policy, its gradients also update the spatial and temporal streams. These gradients encourage the encoder to capture motion inertia, delay, and interaction dynamics, helping the policy anticipate upcoming hand motion from recent history.

### C. League Training

Stable difficulty regulation also depends on diversity in the training hand behaviors. A robot trained against a single hand policy may overspecialize and fail when the pursuit strategy changes. Simultaneous training of the robot and hand creates a different problem because both agents jointly determine the next state and return. Updating both policies together changes each agent's effective environment, violating the stationarity assumption used by PPO. In practice, the interaction can enter a cycle in which the robot exploits the current hand policy, the hand adapts, and the robot loses strategies that were useful against earlier hands.

The training schedule separates the two sides of learning and retains older hand behaviors in a shared pool. During robot training, sampled hand policies are fixed and treated as part of the environment. During hand training, the robot policy is fixed. The robot is trained against a pool P containing the SPC and learned hand policies from previous stages. This pool exposes the robot to direct pursuit as well as more strategic interception patterns. It broadens the distribution of hand behaviors without making the task purely adversarial, since the robot must maintain the ZPD interaction rather than simply maximize escape.

Algorithm 1 summarizes the iterative procedure. Each iteration first trains the robot against hands sampled from the fixed pool. It then trains a new hand policy against the frozen robot, initializes that hand from the previous learned hand when available, and adds the resulting policy to the pool for the next phase of robot training.

**Algorithm 1: Iterative League Training**

**Input:** initial robot policy π_R^(0), scripted pursuit controller π_H^SPC, number of league iterations N.

**Initialize:** pool of hand policies P ← {π_H^SPC}.

1. **for** n = 1, 2, ..., N **do**
2. Train robot policy π_R^(n) with PPO against hand policies sampled from P.
3. Initialize hand policy π_H^(n) from π_H^(n-1) when available.
4. Train π_H^(n) against the frozen robot policy π_R^(n).
5. Update pool of hand policies P ← P ∪ {π_H^(n)}.
6. **end for**

Uniform sampling from the pool of learned hands is inefficient because different hands challenge the current robot to different degrees. Recent episode length provides a simple competitiveness signal. Short episodes usually indicate that the hand quickly catches the microrobot or forces early termination, so the corresponding hand should be sampled more often during the next phase of robot training. For learned hand i, let l_bar_i be its recent mean episode length. Its sampling score is inversely related to this length and is smoothed with a uniform mixture.

$$s_i = \frac{1}{\bar{\ell}_i+\epsilon_{\ell}}, \quad \tilde{p}_i = (1-\mu)\frac{s_i}{\sum_j s_j} + \frac{\mu}{m}. \tag{10}$$

Here, m is the number of learned hand policies, ε_l is a small stabilizing constant, and μ controls the uniform exploration mass. The SPC remains in the pool with a fixed sampling probability, while the probability assigned to learned hands follows (10). This keeps the scripted pursuit behavior available and concentrates the remaining episodes of robot training on learned hands that still expose weaknesses in the current robot policy.

## IV. EXPERIMENTS AND RESULTS

The evaluation combines simulation experiments with a physical deployment test. The simulation experiments measure robustness across hand behaviors and isolate the contribution of temporal encoding of hand history. The physical deployment test evaluates whether the learned policy can run online within the perception and control stack of the UR10 platform and whether the realized motion responds coherently to measured hand movement.

### A. Simulation Protocol and Metrics

The simulation study separates two sources of performance gain in the proposed framework. The league analysis tests whether expanding the pool of hand policies improves robustness across pursuit behaviors. The ablation analysis tests whether temporal encoding of hand motion and the auxiliary dynamics objective improve the learned robot representation. Both analyses use the simulation environment described in Section II, with the robot controlling the microrobot and the hand motion generated by the behavior models described in Section III.

All simulation evaluations used the same planar workspace and the target ZPD band defined in Section II.B. A timestep was counted as within the zone when the distance between the hand and microrobot fell inside the target ZPD band. The simulator and the physical workspace share the same calibrated coordinate convention, so one workspace unit corresponds to 1 cm after calibration and can be compared directly with physical rollout distances. The physical rollout in Section IV.D used the same target ZPD band for zero-shot consistency, not to suggest that this operating setting is clinically optimal for every user.

Time in Zone (TIZ) is the primary metric. It measures the fraction of the full episode horizon for which the distance between the hand and microrobot remains inside the target ZPD band. Episode length is reported in simulation steps and measures how long the interaction remains active before catch, workspace exit, or the maximum horizon. For the league analysis, robustness across learned hand policies is summarized by mean TIZ and by the lowest TIZ among the tested hands. For the network ablation, TIZ is complemented by workspace coverage and mean jerk, which respectively measure task coverage and motion smoothness in the final policy evaluation.

The comparison of training protocols uses an SPC baseline trained only against the stochastic scripted pursuit controller, a single policy baseline trained against one learned hand policy, and a league policy trained through iterative expansion of the hand pool. These robot policies are evaluated against learned hands from different generations and through the scripted, learned policy, and manual mouse interfaces used in Table II. The representation comparison keeps the training setting fixed while changing the policy encoder. The MLP baseline uses instantaneous geometry, the GRU policy adds recent relative hand displacement, and the auxiliary GRU policy adds the objective for future dynamics prediction.

### B. League Training and Robustness Across Hand Policies

The league evaluation asks whether an expanding pool of hand policies improves robustness across pursuit behaviors, rather than only improving performance against the most recent hand policy. After each robot generation is trained, it is evaluated against all learned hand generations in the pool. This separates generalization across hand behaviors from progress against a single training opponent. In Fig. 3(a), each row fixes one robot generation and each column fixes one learned hand generation. Early robot generations perform well only for limited parts of the hand pool, whereas later generations maintain stronger TIZ over a broader set of learned hands, suggesting reduced specialization. Fig. 3(b) summarizes the same matrix through mean TIZ and the lowest TIZ across tested hands. The mean curve reflects typical performance, and the lower curve tracks the most difficult learned hand for each robot generation. Both improve from early to final generations despite local fluctuations.

**Fig. 3. Validation of robot policies trained through league across learned hand generations using the target ZPD band. (a) TIZ matrix obtained by evaluating each robot generation against each learned hand generation. Rows correspond to robot generations, and columns correspond to learned hand generations. (b) Mean TIZ and lowest TIZ across the learned hand test policies for each robot generation.**

Table II evaluates whether this robustness extends beyond the pool of learned hands used in Fig. 3. The final SPC baseline, single policy baseline, and league policy are tested against the stochastic scripted pursuit controller, a learned hand policy, and manual mouse pursuit. The manual mouse condition was completed by one healthy operator using the mouse interface to pursue and catch the simulated microrobot as quickly as possible. It introduces human reaction time and voluntary pursuit strategy, but it does not reproduce the biomechanics or perception noise of real hand tracking. It is therefore treated as a human-in-the-loop stress test rather than clinical validation. Higher TIZ and longer episode length indicate more sustained difficulty regulation.

**TABLE II: POLICY COMPARISON ACROSS THREE HAND TEST SETS**

| Robot policy | Test set | TIZ | Length |
| --- | --- | ---: | ---: |
| SPC baseline | SPC | 0.31 ± 0.06 | 52 ± 8 |
| SPC baseline | Learned hand | 0.19 ± 0.04 | 31 ± 5 |
| SPC baseline | Manual mouse | 0.14 ± 0.03 | 33 ± 9 |
| Single policy | SPC | 0.11 ± 0.02 | 25 ± 4 |
| Single policy | Learned hand | 0.49 ± 0.07 | 70 ± 7 |
| Single policy | Manual mouse | 0.19 ± 0.05 | 40 ± 11 |
| League | SPC | 0.38 ± 0.06 | 67 ± 8 |
| League | Learned hand | 0.48 ± 0.07 | 64 ± 8 |
| League | Manual mouse | 0.49 ± 0.09 | 74 ± 13 |

Values are means ± approximate 95% confidence intervals estimated across evaluation episodes. SPC and learned hand tests used 100 episodes per condition, and manual mouse tests used 20 episodes per robot policy. TIZ denotes the fraction of the full episode horizon spent in the ZPD, and Length denotes episode length in simulation steps.

The two baselines without league training show complementary forms of specialization. The SPC baseline remains most suitable for the scripted pursuit controller but degrades under learned and manual pursuit. The single policy baseline performs well against the learned hand policy but loses robustness when the pursuit behavior changes to scripted or manual control. The league policy gives the most balanced profile across all three test sets, preserving performance against the learned hand while improving the scripted and manual pursuit cases. This balance is the desired behavior for adaptive difficulty regulation, because hand behavior can change across controllers, sessions, and voluntary strategies.

### C. Network Ablation and Auxiliary Dynamics Analysis

The league results evaluate the diversity of hand behaviors used during training, while the network ablation tests whether the robot policy can use temporal information about hand motion once that diversity is present. The MLP baseline uses the instantaneous geometric state only, the GRU policy adds a history of relative displacement over 16 frames, and the auxiliary GRU further adds the dynamics prediction head described in Section III.B. The training curves in Fig. 4(a) and Fig. 4(b) show that temporal history improves optimization and interaction persistence. The GRU policies reach higher episode reward and longer episode length than the MLP baseline, indicating that current geometry alone is insufficient for stable difficulty regulation when hand motion changes over time. The auxiliary GRU gives the strongest training curve, suggesting that the prediction loss helps the recurrent stream form a more useful motion representation.

The auxiliary prediction examples in Fig. 4(c) provide a qualitative check on this representation. The predicted displacement over the short horizon generally follows the direction and curvature of the observed future hand motion rather than only reconstructing the current position. The final normalized metrics in Fig. 4(d) summarize the control effect of these representation changes. Recurrent history improves time within the ZPD and workspace coverage relative to the MLP baseline. Adding the auxiliary prediction objective further improves these performance metrics while reducing mean jerk. Together, the ablation indicates that temporal hand history and auxiliary dynamics prediction contribute to smoother and more reliable difficulty regulation.

**Fig. 4. Network ablation for temporal encoding and auxiliary dynamics prediction. (a) Smoothed episode reward during training for the MLP, GRU, and auxiliary GRU policies. (b) Smoothed episode length during training for the same policies. (c) Representative predictions from the auxiliary head over a short horizon compared with observed future hand motion. (d) Final performance metrics normalized to the MLP baseline; higher values are better for time within the ZPD and workspace coverage, whereas lower values are better for mean jerk.**

### D. Online Physical Deployment

The policy trained in simulation was deployed on the UR10 microrobot platform to evaluate zero-shot transfer with real hand input. In each rollout, a healthy pilot pursued the passive microrobot on the acrylic workspace while the UR10 moved the magnetic actuator beneath the surface. The policy received no further training on the physical platform, and the dataset contains nine zero-shot rollouts from one pilot session. The camera and UR10 command loops provided real-time control, and policy inference was fast enough to support online interaction with continuous feedback.

The deployment stack preserved the simulation observation format. Policy inputs were constructed from calibrated camera measurements of the hand and microrobot, features describing workspace boundaries, the previous executed robot command, and the recent history of relative hand displacement, matching the categories in (6) and (7). UR10 telemetry and end-effector pose were not provided to the policy; RTDE pose readback was used only for Cartesian command execution and monitoring over Ethernet. The camera tracking pipeline handled perception and calibration through YOLO microrobot detection, MediaPipe hand tracking, and a homography that mapped image coordinates to workspace coordinates. During short gaps in hand tracking, the controller extrapolated motion from the last measured hand velocity. Safety constraints included workspace clipping, limits on command increments, automatic stopping after catch, UR10 servoStop on shutdown or interruption, and the standard hardware emergency stop available during testing.

Fig. 5 presents one representative zero-shot physical rollout. The image sequence in Fig. 5(a) and the dynamic traces in Fig. 5(b) come from the same trial and share the same time axis. The selected frames cover the initial approach, steady pursuit, faster and slower hand movements, and the end of the rollout. They show that camera perception, hand pursuit, microrobot motion, and UR10 actuation operated within one feedback loop. The distance trace remains near the target ZPD band for a substantial portion of the trial and returns toward the target ZPD band after deviations. The controller therefore neither simply moves the microrobot away from the hand nor keeps it static until catch. The profiles of hand speed and robot speed support the same interpretation: changes in measured hand speed are reflected in the distance trace and in the robot response. The deployment evaluates the complete loop connecting camera, policy, and robot rather than offline policy inference alone. It shows that the controller trained in simulation can sustain an online interaction between the hand and microrobot on the UR10 platform.

**Fig. 5. Zero-shot physical deployment on the UR10 microrobot platform. (a) Key deployment frames from a representative rollout among nine physical trials. (b) Smoothed distance between the hand and microrobot, target ZPD band, and measured hand and robot speeds from the same rollout.**
