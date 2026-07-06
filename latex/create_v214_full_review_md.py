from pathlib import Path
from shutil import copy2
from docx import Document
from docx.document import Document as _Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P

src = Path("manuscripts/methodology_ieee_v2.12_fig1_corrected.docx")
out = Path("manuscripts/methodology_ieee_v2.14_full_prose_flow_review.md")

doc = Document(src)

replacements = {
    "We introduce a magnetically actuated, non-contact rehabilitation system designed to reduce the physical risks associated with rigid robotic interaction while preserving an engaging motor-training task. This section first describes the hardware architecture and then formulates the control problem as a reinforcement learning (RL) task.":
    "We introduce a magnetically actuated, non-contact rehabilitation system that preserves an active motor-training task while avoiding rigid physical coupling between the robot and the patient. The control problem is formulated as a reinforcement learning (RL) task in which the robot regulates interaction difficulty through the motion of a magnetic microrobot.",

    "The platform improves safety by physically separating the robotic actuator from the patient. As shown in Fig. X, the system consists of three layers. The actuation layer places a UR10 robotic arm beneath the workspace, with a permanent magnet mounted on the end-effector to drive the microrobot above. The interaction layer is a 5-mm acrylic sheet that supports the unpowered magnetic microrobot and serves as a barrier between the arm and the patient. The perception layer uses an overhead RGB camera connected to a host computer, which also communicates with the UR10 over Ethernet. At each control cycle, the computer acquires a camera frame, estimates the positions of the microrobot and the patient's hand, and sends target pose commands to the arm.":
    "The platform separates the robotic actuator from the patient through a three-layer layout. In the actuation layer, a UR10 robotic arm is placed beneath the workspace, with a permanent magnet mounted on the end-effector to drive the microrobot above. In the interaction layer, a 5-mm acrylic sheet supports the unpowered magnetic microrobot and forms a physical barrier between the arm and the patient. In the perception layer, an overhead RGB camera sends images to a host computer, which estimates the microrobot and hand positions and transmits target pose commands to the UR10 over Ethernet.",

    "This physical separation is central to the safety design. The acrylic interface reduces the risk of blunt impact, excessive traction, and crushing injuries that may occur in direct-contact rehabilitation devices. The hardware is also simple and reproducible: it requires a collaborative robot, a camera, and a permanent magnet, without wearable components, force sensors, or custom actuation modules. Because the microrobot is unpowered and driven passively by the magnetic field, it does not require onboard electronics or batteries. This simplicity facilitates setup and maintenance, and it provides an additional safety margin during the iterative development of data-driven controllers.":
    "The same physical layout serves two design goals: safety and reproducibility. The acrylic interface reduces the risk of blunt impact, excessive traction, and crushing injuries that may occur in direct-contact rehabilitation devices. At the same time, the hardware requires only a collaborative robot, a camera, and a permanent magnet; it does not rely on wearable components, force sensors, onboard batteries, or custom actuators. Passive magnetic actuation keeps the patient-side device lightweight and unpowered, which simplifies setup while limiting the consequences of controller errors during development.",

    "We formulate the rehabilitation task as a pursuit-evasion game on a flat workspace. The patient sits in front of the acrylic surface and uses one hand to chase the magnetic microrobot, which is controlled by the robot arm below. The goal is to bring the hand within a small contact radius of the microrobot. Unlike static reaching exercises, the microrobot actively evades the patient's hand, requiring continuous visuomotor tracking, trajectory prediction, and real-time motor planning. Such goal-directed tracking can impose sustained cognitive and physical demands beyond point-to-point reaching and is relevant to motor recovery [X]. To maximize therapeutic benefit, task difficulty must adapt to the patient's current ability. We formalize this requirement using the Zone of Proximal Development (ZPD), which describes tasks that remain challenging but achievable with support [X]. In our setting, the ZPD corresponds to a target distance range between the hand and the microrobot. The robot should neither escape completely nor yield passively; instead, it should keep the interaction within this range so that the patient remains engaged without becoming fatigued or frustrated. This objective is challenging because patient movement speed, strategy, delay, and motor noise can vary over time.":
    "We formulate the rehabilitation task as a pursuit-evasion game on a flat workspace. The patient sits in front of the acrylic surface and uses one hand to chase the magnetic microrobot, which is controlled by the robot arm below. The goal is to bring the hand within a small contact radius of the microrobot. Unlike static reaching exercises, the microrobot actively evades the patient's hand, requiring continuous visuomotor tracking, trajectory prediction, and real-time motor planning. Goal-directed tracking targets sustained visuomotor coordination rather than isolated point-to-point movement [X].\n\nTo keep the task challenging but achievable, we define a Zone of Proximal Development (ZPD) as a target distance range between the hand and the microrobot [X]. The robot should neither escape completely nor yield passively; it should keep the interaction inside this range so that the patient remains engaged without becoming fatigued or frustrated. Patient speed, strategy, delay, and motor noise can all change during interaction, making fixed difficulty settings inadequate.",

    "Conventional controllers are not well suited to this setting. Methods based on artificial potential fields or rule-based impedance control usually rely on hand-crafted responses to instantaneous state feedback. They therefore have limited ability to anticipate higher-level patient strategies, such as interception, or to adapt to heterogeneous movement patterns. Their performance may degrade further under tremor, delayed response, or bradykinesia because they do not explicitly distinguish purposeful movement from involuntary perturbations. RL provides a natural framework for learning such adaptive behavior through repeated interaction. In our formulation, the robot learns an evasion policy that is rewarded for keeping the hand-microrobot distance within the therapeutic range and penalized for unsafe or uninformative behavior, such as escaping too far or being caught too easily. By conditioning the policy on temporal observations, the controller can infer patient intent from movement history rather than reacting only to instantaneous position error. The design choices in Section IV--domain randomization, temporal encoding, and league training--address these sources of patient variability.":
    "Conventional controllers are not well suited to this setting. Methods based on artificial potential fields or rule-based impedance control usually rely on hand-crafted responses to instantaneous state feedback. They have limited ability to anticipate higher-level patient strategies, such as interception, or to adapt to heterogeneous movement patterns. Performance can degrade further under tremor, delayed response, or bradykinesia because purposeful movement and involuntary perturbation are not explicitly separated.\n\nReinforcement learning is useful here because the controller can optimize a long-horizon ZPD objective under variable patient behavior, rather than following a fixed local response rule. In our formulation, the robot learns an evasion policy that is rewarded for keeping the hand-microrobot distance within the therapeutic range and penalized for unsafe or uninformative behavior, such as escaping too far or being caught too easily. Temporal observations allow the policy to infer patient intent from movement history instead of reacting only to instantaneous position error. Section IV introduces domain randomization, temporal encoding, and league training as the three mechanisms used to handle patient variability.",

    "This section describes the training framework. We first introduce Cognitive-Motor Decoupled Domain Randomization (CMD-DR), which separates patient movement strategy from motor execution limitations. We then present a dual-stream encoder with an auxiliary future-dynamics head for extracting motion information from noisy temporal observations. Finally, we describe a league-training procedure designed to improve robustness against a growing pool of hand behaviors.":
    "The training framework combines three components. Cognitive-Motor Decoupled Domain Randomization (CMD-DR) separates patient movement strategy from motor execution limitations. A dual-stream encoder extracts geometric and temporal motion information from noisy observations, and an auxiliary future-dynamics head provides an additional training signal for the temporal representation. Iterative league training then exposes the robot to a growing pool of hand behaviors.",

    "A single frame is often insufficient for inferring patient intent. Observed hand motion may reflect sensing noise, motor delay, tremor-like perturbations, or slow response, rather than deliberate pursuit. We therefore use a dual-stream encoder with an auxiliary prediction task, as shown in Fig. X.":
    "A single frame is often insufficient for inferring patient intent. Observed hand motion may reflect sensing noise, motor delay, tremor-like perturbations, or slow response, rather than deliberate pursuit. We therefore use a dual-stream encoder with an auxiliary prediction task.",

    "The temporal buffer h_{t-T:t} contains the T = 16 most recent relative hand displacement vectors, forming a 32-dimensional sequence. This buffer captures velocity, acceleration, and repeated movement patterns that are unavailable from the current position alone. The scalar vector is processed by a Multi-Layer Perceptron (MLP) to extract geometric information. In parallel, the temporal buffer is processed by a Gated Recurrent Unit (GRU) sequence encoder. Because the GRU operates on relative displacements rather than absolute positions, it emphasizes movement direction, speed, acceleration, and oscillation. The two stream outputs are concatenated and passed through a fusion MLP to produce a shared representation for policy and value prediction.":
    "The temporal buffer h_{t-T:t} contains the T = 16 most recent relative hand displacement vectors, forming a 32-dimensional sequence. It captures velocity, acceleration, and repeated movement patterns that are unavailable from the current position alone. Using relative displacements rather than absolute positions makes the history representation translation-invariant and emphasizes movement direction, speed, acceleration, and oscillation.\n\nThe scalar vector and temporal buffer are encoded in separate streams. A Multi-Layer Perceptron (MLP) extracts geometric information from the scalar state, while a Gated Recurrent Unit (GRU) processes the displacement sequence. The two stream outputs are concatenated and passed through a fusion MLP to produce the shared representation used by the policy and value heads.",

    "The ZPD-based reward provides a control signal, but it does not directly supervise how the encoder should represent patient motion. Learning this representation only through policy reward can therefore be slow and unstable. To provide a denser signal, we attach a future-dynamics head to the fused representation. This head predicts the patient's relative displacement over an eight-step horizon and estimates near-catch risk. Because future displacements are already available from the rollout, the auxiliary targets require no additional annotation. The trajectory loss is defined as:":
    "The ZPD-based reward provides a control signal, but it does not directly supervise how the encoder should represent patient motion. Learning this representation only through policy reward can therefore be slow and unstable. To provide a denser signal, we attach a future-dynamics head to the fused representation. The head predicts the patient's relative displacement over an eight-step horizon and estimates near-catch risk. Because future displacements are already available from the rollout, the auxiliary targets require no additional annotation. The trajectory loss is defined as:",

    "where lambda_traj and lambda_risk control the auxiliary loss weights. Because the future-dynamics head shares the encoder with the policy, its gradients also update the spatial and temporal streams. This encourages the encoder to capture patient inertia, delay, and interaction dynamics, enabling the policy to anticipate near-future hand motion rather than reacting only to the current hand-microrobot distance.":
    "where lambda_traj and lambda_risk control the auxiliary loss weights. Because the future-dynamics head shares the encoder with the policy, its gradients also update the spatial and temporal streams. Shared auxiliary gradients encourage the encoder to capture patient inertia, delay, and interaction dynamics, enabling the policy to anticipate near-future hand motion rather than reacting only to the current hand-microrobot distance.",

    "where w_i denotes the normalized episode length against opponent i, k controls the sharpness of the distribution, and eta defines the tolerance band around the target value of 0.5. This sampling rule gives higher probability to hand policies that produce intermediate episode lengths. In this way, training effort is automatically concentrated on the most informative opponents without requiring a manually specified curriculum.":
    "where w_i denotes the normalized episode length against opponent i, k controls the sharpness of the distribution, and eta defines the tolerance band around the target value of 0.5. The resulting distribution assigns higher probability to hand policies that produce intermediate episode lengths, concentrating training on informative opponents without requiring a manually specified curriculum.",

    "One possible risk of league training is that the robot may learn to exploit artificial patterns in the simulated hand policies. Such behavior would reduce the relevance of the learned policy for real rehabilitation. To reduce this risk, a fixed probability p_s is always assigned to the heuristic scripted hand policy. This ensures that the robot remains exposed to simple, noisy pursuit behavior, which is closer to the behavior expected from real patients.":
    "One possible risk of league training is that the robot may learn to exploit artificial patterns in the simulated hand policies. Such behavior would reduce the relevance of the learned policy for real rehabilitation. To reduce this risk, a fixed probability p_s is always assigned to the heuristic scripted hand policy, keeping the robot exposed to simple, noisy pursuit behavior closer to what may be expected from real patients.",

    "The experiments evaluate the proposed framework from complementary perspectives rather than following the methodology section mechanically. We first define the common simulation setup, training protocols, and evaluation metrics. The main evaluation then examines whether league training improves robustness across learned hand behaviors and how the opponent-sampling process evolves during training. We next analyze the role of temporal interaction history and the auxiliary future-dynamics head through a network ablation. The section closes with the real-world deployment pipeline that connects the simulation policy to the UR10 platform.":
    "The experiments evaluate three questions: whether league training improves robustness across learned hand behaviors, whether temporal motion history is necessary for adaptive difficulty regulation, and whether the learned simulation policy can be connected to the physical UR10 platform through a fixed-rate control pipeline.",

    "The primary evaluation metric is Time-In-Zone (TIS), the fraction of the episode horizon during which the interaction remains inside the ZPD band. We also report ZPD coverage, episode length, catch rate, and empirical robustness across the learned hand pool. These metrics capture different aspects of the same rehabilitation objective: TIS measures sustained therapeutic interaction over the full horizon, ZPD coverage measures the quality of the realized portion of an episode, episode length captures survival, and catch rate reflects overly easy interactions in which the microrobot is caught too quickly.":
    "The primary evaluation metric is Time-In-Zone (TIS), the fraction of the episode horizon during which the interaction remains inside the ZPD band. We also report ZPD coverage, episode length, catch rate, and empirical robustness across the learned hand pool. Together, the metrics capture different aspects of the same rehabilitation objective: TIS measures sustained therapeutic interaction over the full horizon, ZPD coverage measures the quality of the realized portion of an episode, episode length captures survival, and catch rate reflects overly easy interactions in which the microrobot is caught too quickly.",

    "This evaluation tests whether opponent-pool training reduces the brittle specialization that can occur when the robot is optimized against a single hand model. The league policy is trained for ten generations while the hand pool is expanded with learned hand policies. During robot training, the opponent identity is not given as an input; the robot must infer the current hand behavior from the spatial state and recent displacement history. This design makes the evaluation closer to rehabilitation use, where patient capability is observed through movement rather than provided as a label.":
    "The league evaluation tests whether opponent-pool training reduces the brittle specialization that can occur when the robot is optimized against a single hand model. The robot is trained for ten generations while the hand pool is expanded with learned hand policies. Opponent identity is not provided to the robot; hand behavior must be inferred from the spatial state and recent displacement history, as would be required when patient capability is observed through movement rather than given as a label.",

    "Fig. 1 summarizes the league result from both evaluation and training-process perspectives. Panel (a) gives the cross-iteration validation matrix: each row is a robot generation, each column is a learned hand generation, and each cell reports TIS for that robot-hand pairing. This panel shows how performance changes across the full interaction matrix rather than only for the final policy. The stronger values in the first hand column indicate that early learned hands remain easier to regulate, while later columns reveal the harder cases that drive the need for broader opponent coverage.":
    "Fig. 1(a) evaluates every robot generation against every learned hand generation, producing a cross-iteration TIS matrix rather than a single final score. The matrix exposes both sides of the league: later robot generations generally become more capable, while later learned hands introduce harder pursuit behaviors. Early learned hands remain easier to regulate, but the harder columns reveal why a robot trained on a narrow opponent set can appear competent while still failing on other hand strategies.",

    "Panel (b) reduces the same validation matrix into generation-level robustness curves. The mean TIS measures average performance across learned hands, the worst-hand TIS identifies the most difficult opponent in the pool, and CVaR20 summarizes the lower tail rather than a single extreme case. The upward trend toward the final generation indicates that league training improves not only the average interaction quality but also the less favorable part of the opponent distribution.":
    "The aggregate views in Fig. 1(b) and Fig. 1(c) separate average performance from difficult-case robustness. Mean TIS increases across generations, but the more important trend is the improvement in worst-hand TIS and CVaR20, which measure the lower tail of the learned-hand distribution. The robustness frontier makes the same point geometrically: later robot generations move toward policies that improve average ZPD regulation without sacrificing the hardest hand cases.",

    "Panel (c) presents the robustness frontier by plotting each robot generation in the space of mean TIS and worst-hand TIS, with color indicating CVaR20. This view separates policies that improve average behavior from those that also improve difficult-case behavior. In the rehabilitation setting, this distinction matters because a controller that performs well only for easy virtual patients may still fail to maintain useful interaction for patients with more challenging movement patterns.":
    "Fig. 1(d) and Fig. 1(e) connect final behavior to the training process. The final-generation failure decomposition separates episodes that are too close from those that are too far, showing that different hand generations fail in different ways even when ZPD coverage remains useful. PFSP sampling snapshots show the corresponding training mechanism: sampling probability shifts toward opponents that remain informative, while the probability floor keeps older hand policies in the pool. The league therefore improves robustness not by optimizing one final opponent, but by repeatedly reallocating training pressure across a changing set of hand behaviors.",

    "We further test whether the learned robustness transfers across hand-controller types. For this comparison, the final scripted-only, single-agent, and league-trained robot policies are each evaluated against two hand mechanisms: the stochastic scripted pursuit controller and a learned hand agent. The table entries report TIS, ZPD coverage, episode length, and catch rate in that order. Mouse-controlled human-in-the-loop testing is kept separate because it introduces human reaction time and voluntary strategy rather than another automated controller.":
    "We further test whether the learned robustness transfers across hand-controller types. For this comparison, the final scripted-only, single-agent, and league-trained robot policies are each evaluated against two hand mechanisms: the stochastic scripted pursuit controller and a learned hand policy. The table entries report TIS, ZPD coverage, episode length, and catch rate in that order. Mouse-controlled human-in-the-loop testing is kept separate because it introduces human reaction time and voluntary strategy rather than another automated controller.",

    "Table II shows complementary failure modes for the two non-league baselines. The scripted-only robot remains competitive on the scripted controller but is caught frequently by the learned hand agent, indicating that rule-based pursuit during training does not cover learned strategic behavior. The single-agent robot shows the reverse tendency: it performs well against the learned hand but degrades on the scripted controller. The league-trained robot does not dominate every metric in every column, but it avoids the severe collapse observed in the baselines and provides the most balanced behavior across both automated test conditions. This balance is the relevant outcome for an adaptive rehabilitation controller, whose goal is not to overfit to one hand model but to maintain therapeutic interaction as patient behavior changes.":
    "Table II shows complementary failure modes for the two non-league baselines. The scripted-only robot remains competitive on the scripted controller but is caught frequently by the learned hand policy, indicating that rule-based pursuit during training does not cover learned strategic behavior. The single-agent robot shows the reverse tendency: it performs well against the learned hand policy but degrades on the scripted controller. The league-trained robot does not dominate every metric in every column, but it avoids the severe collapse observed in the baselines and provides the most balanced behavior across both automated test conditions. For adaptive rehabilitation, balanced robustness is more important than overfitting to a single hand model because patient behavior can shift across sessions and within an episode.",

    "The upper panels of Fig. 2 show that the main performance gain appears when temporal interaction history is introduced. Both recurrent policies learn longer and more rewarding interactions than the MLP baseline. This suggests that the robot benefits from observing how the hand has been moving, not only where it is at the current instant. In this setting, the GRU-only and GRU+Aux policies follow similar aggregate learning trends, indicating that temporal sequence modeling is the dominant contributor to control performance.":
    "The upper panels of Fig. 2 show that the main performance gain appears when temporal interaction history is introduced. Both recurrent policies learn longer and more rewarding interactions than the MLP baseline, indicating that the robot benefits from observing how the hand has been moving, not only where it is at the current instant. In this setting, the GRU-only and GRU+Aux policies follow similar aggregate learning trends, so temporal sequence modeling is the dominant contributor to control performance.",

    "The lower panels of Fig. 2 provide a complementary interpretation of the auxiliary head. Rather than treating the auxiliary task as a separate source of reward improvement, we use it to inspect whether the temporal encoder has learned predictive structure in hand motion. The predicted trajectories capture short-horizon movement direction and diverge gradually at longer horizons, consistent with the uncertainty of multi-step prediction in an interactive task. Thus, the auxiliary head mainly serves to encourage and expose a temporal representation of patient dynamics.":
    "The lower panels of Fig. 2 provide a complementary interpretation of the auxiliary head. Rather than treating the auxiliary task as a separate source of reward improvement, we use it to inspect whether the temporal encoder has learned predictive structure in hand motion. The predicted trajectories capture short-horizon movement direction and diverge gradually at longer horizons, consistent with the uncertainty of multi-step prediction in an interactive task. In this run, the auxiliary head mainly serves to encourage and expose a temporal representation of patient dynamics.",

    "The final part of the experiment section describes how the simulation policy is connected to the physical UR10 platform. This pipeline is presented as an implementation bridge rather than as a paired sim-to-real benchmark, because real hand kinematics, camera latency, magnetic actuation, and safety constraints are not fully represented in the virtual training environment.":
    "The final part of the experiment section describes how the simulation policy is connected to the physical UR10 platform. The pipeline is best interpreted as an implementation bridge rather than a paired sim-to-real benchmark, because real hand kinematics, camera latency, magnetic actuation, and safety constraints are not fully represented in the virtual training environment.",
}

remove_texts = {
    "Panel (d) decomposes the final robot generation across test hands into too-close rate, too-far rate, and ZPD coverage. The stacked bars distinguish two failure modes: interactions that become too easy because the hand catches the microrobot, and interactions that become too difficult because the microrobot remains too far away. The ZPD-coverage curve shows that the final league policy keeps a meaningful portion of each episode in the therapeutic band even when the type of failure varies across hand generations.",
    "Panel (e) shows snapshots of the PFSP sampling probabilities during selected training iterations. The curves indicate that opponent sampling is not uniform throughout training; probability mass shifts among learned hands as their relative difficulty changes. This supports the training mechanism behind the robustness result: the robot is repeatedly exposed to opponents that remain informative, while the probability floor prevents older opponents from disappearing entirely from the training distribution.",
}


def iter_block_items(parent):
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._tc
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def para_to_md(text):
    text = text.strip()
    if not text:
        return []
    if text in remove_texts:
        return []
    text = replacements.get(text, text)
    parts = []
    for piece in text.split("\n\n"):
        piece = piece.strip()
        if not piece:
            continue
        if piece.startswith(("I. ", "II. ", "III. ", "IV. ", "V. ", "VI. ", "VII. ", "VIII. ")):
            parts.append(f"## {piece}")
        elif len(piece) > 3 and piece[1:3] == ". " and piece[0].isalpha() and piece[0].isupper():
            parts.append(f"### {piece}")
        elif piece.startswith("Fig. "):
            parts.append(f"**{piece}**")
        elif piece.startswith("TABLE"):
            parts.append(f"**{piece}**")
        else:
            parts.append(piece)
    return parts


def table_to_md(table):
    rows = []
    for row in table.rows:
        rows.append([cell.text.strip().replace("\n", " ") for cell in row.cells])
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = []
    lines.append("| " + " | ".join(rows[0]) + " |")
    lines.append("| " + " | ".join(["---"] * width) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return ["\n".join(lines)]

lines = [
    "# methodology_ieee_v2.14_full_prose_flow_review",
    "",
    "Complete Markdown review draft generated from `methodology_ieee_v2.12_fig1_corrected.docx`. Unchanged sections are preserved; targeted prose-flow revisions are applied inline for review before updating the Word manuscript.",
    "",
]
for block in iter_block_items(doc):
    if isinstance(block, Paragraph):
        lines.extend(para_to_md(block.text))
    elif isinstance(block, Table):
        lines.extend(table_to_md(block))
    if lines and lines[-1] != "":
        lines.append("")

out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
print(out.as_posix())
print("paragraph_blocks", sum(1 for b in iter_block_items(doc) if isinstance(b, Paragraph)))
print("tables", sum(1 for b in iter_block_items(doc) if isinstance(b, Table)))
text = out.read_text(encoding="utf-8")
print("has_league_training", "### C. League Training" in text)
print("has_pfsp", "Prioritized Fictitious Self-Play" in text or "PFSP" in text)
print("has_real_world", "### D. Real-World Deployment Pipeline" in text)
print("has_table_ii", "TABLE II" in text)
print("panel_paragraph_count", sum(1 for line in text.splitlines() if line.startswith("Panel ")))
