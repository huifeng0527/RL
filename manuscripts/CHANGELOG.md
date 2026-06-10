# Methodology Paper Changelog

## v1.8 -> v1.9 (2026-06-03)
Comprehensive rewrite for IEEE style compliance. Four changes:

1. **Inline formulas extracted to standalone equations**: The observation decomposition (o_t = [s_t ; h_{t-T:t}]) is now a standalone equation with a following explanatory paragraph. All inline math (u_t, d_hat_{t+1}, etc.) replaced with descriptive prose.

2. **Vocabulary simplified**: Replaced complex phrasing ("intrinsic safety achieved through strict physical decoupling" → "safety by physically separating"), reduced sentence length, removed jargon where plain language suffices.

3. **Small paragraphs merged**: Eliminated 18 empty/single-sentence paragraphs. Merged Section III intro into Hardware Platform intro, merged MDP formulation into Problem Formulation, merged four OOD test protocols into one paragraph, merged vision/control thread descriptions into one paragraph, combined auxiliary loss explanation with its equations.

4. **Other IEEE standards**: Consistent "Fig. X" references, equations introduced with proper lead-in sentences, no orphan headings, balanced paragraph lengths throughout.

## v1.6 -> v1.8 (2026-06-03)
Five targeted fixes based on manuscript review:

1. **Issue 1 — Observation space formula (Section IV-B)**: Added explicit 12-dimensional observation decomposition formula after the IV-B intro paragraph: s_t = [p_R, p_H, d(R,H), b_N, b_S, b_E, b_W, s, a_{t-1}] in R^12, plus the 32-dim temporal buffer description.

2. **Issue 5 — TABLE III framework (Section VI-B)**: Inserted a 6x5 table with columns [Protocol, Metric, Baseline A (Scripted Only), Baseline B (Single RL), Ours (League)] and 5 data rows (Sluggish/Avg Distance, Spasm/ZPD Coverage, Unseen RL/ZPD Coverage, Human Mouse/Avg Distance, Human Mouse/ZPD Coverage). Data cells marked TBD.

3. **Issue 6 — Tremor description reconciliation (Section IV-B)**: Replaced "repetitive oscillatory motions (resembling tremors) and slow movement patterns (resembling bradykinesia)" with language consistent with the Gaussian noise model from v1.2: now references "Gaussian observation noise and neural delay" from the biomechanical filter in Section IV-A.

4. **Issue 10 — PFSP metric unification (Section IV-C)**: Changed "fraction of timesteps the robot maintains the hand-robot distance within the ZPD" to "robot's average episode length against each pool member as a proxy for competitive balance". Updated "50% ZPD-coverage target" to "50% episode-length target" to match the inverse-survival-probability implementation.

5. **Issue 11 — Unicode corruption fix (Section IV-A)**: Replaced corrupted Unicode delta symbols (δ, zero-width spaces, tilde variants) with clean ASCII: "delta ~ U(delta_min, delta_max)" and "t - delta".

## v1.6 -> v1.7 (2026-06-02)
Polished the existing manuscript draft while preserving the original section structure and technical content.

1. **Language polish throughout**: Tightened IEEE-style academic phrasing, reduced repetition, improved transitions, and standardized wording for the hardware platform, ZPD formulation, domain randomization, dual-stream encoder, league training, OOD evaluation, and sim-to-real transfer sections.

2. **Terminology and formatting consistency**: Normalized expressions such as "Fig. X", "5 mm", "hand-position estimation", "domain-randomization", "feature-extractor", and "zero-shot generalization"; replaced fragile Greek/math glyphs with stable ASCII notation where Word rendering had produced corrupted symbols.

3. **Layout cleanup**: Saved the polished draft as `methodology_ieee_v1.7.docx`, added standard section properties to the DOCX, and adjusted pagination so Algorithm 1 and TABLE III render as coherent blocks.

## v1.5 → v1.6 (2026-06-01)
Major rewrite of Section VI based on huifeng's feedback. Three changes:

1. **A. Network Architecture Ablation**: Removed ZPD-coverage/ZPD-Steps table. Now only references reward and episode length curves (Fig. X). Updated narrative: MLP-Only converges quickly to a low level; aux head doesn't change convergence speed but raises final performance.

2. **B. Zero-Shot OOD Generalization**: Restructured narrative to focus on validating iterative league training (not just "testing generalization"). Added Human Mouse Hand test (4th protocol). Removed numerical result placeholders — will fill after user provides data.

3. **C. Sim-to-Real Transfer**: Moved from Section IV-D (Methodology) to Section VI-C (Experiments). Rewritten as experimental validation rather than architectural description. Removed Section IV-D entirely.

**Note**: v1.5 was locked in Word, saved as v1.6.

## v1.4 (2026-06-01)
Based on 3 new annotations from huifeng:

1. **Comment 0 — Shorten long sentence**: The rehabilitation objective sentence was too long and complex. Split into two shorter sentences: "The robot does not seek to defeat an opponent. Instead, it must maintain therapeutic efficacy across the full spectrum of patient capability, from severely impaired to high-functioning."

2. **Comment 1 — Remove colon, shorten warm-starting**: "Warm-starting is critical: it preserves..." → "Warm-starting is critical because it preserves... Each new hand can thus build upon a richer behavioral repertoire rather than starting from scratch."

3. **Comment 2 — Replace win rate with ZPD-coverage metric**: The task has no "win rate" — instead, performance is measured by the fraction of timesteps the robot keeps the distance within the ZPD. Changed: "empirical win rate" → "fraction of timesteps within the ZPD"; "win-rate prioritized" → "ZPD-coverage prioritized"; "50% win-rate target" → "50% ZPD-coverage target"; "defeats roughly half the time" → "spends roughly half the interaction time within the ZPD".

## v1.3 (2026-06-01)
Based on 2 new annotations from huifeng:

1. **Comment 0 — Remove 3 equations + inline symbols**: Removed the observation decomposition equation (o_t = [s_t ; h_{t-T:t}]), LSTM equation (e_t^k, c_t = g_psi(...)), and fusion equation (z_t = f_fus(...)). Rewrote all descriptive paragraphs without symbolic notation — spatial stream and temporal stream described in plain prose.

2. **Comment 1 — Remove "However" + simplify "sparse rewards"**: "However, sparse reinforcement learning rewards" was not a转折 relationship. Changed to "Sparse rewards" as a direct subject. Also simplified "sparse RL rewards" → "sparse rewards" in the closing paragraph.

## v1.2 (2026-06-01)
Based on 6 new annotations from huifeng:

1. **Comment 0 — Problem Formulation transition**: "Maintaining this adaptive difficulty is non-trivial." was too abrupt. Added "However," at the beginning for better flow from the ZPD paragraph.

2. **Comment 1 — Agent introduction**: "The agent observes..." introduced "agent" without context after discussing RL formulation. Changed to "Under this formulation, the robot policy observes..." to connect back to the MDP discussion.

3. **Comment 2 — Methodology intro de-listification**: Removed the list-style "four interdependent challenges" summary (AI-sounding). Replaced with a single flowing sentence summarizing what the RL approach achieves.

4. **Comment 3 — DR section restructuring**: Added Gymnasium environment context before discussing domain randomization. Now introduces the two entities (robot as learning subject, hand as patient emulator) and explains why the hand must be realistic to avoid sim-to-real gap.

5. **Comment 4 — Tremor → Gaussian noise**: Replaced Parkinsonian tremor model (narrow-band harmonics tau_t = A * sin/cos) with Gaussian observation noise (x_t' = x_t + epsilon, epsilon ~ N(0, sigma^2)) to simulate visual extraction error instead of clinical tremor.

6. **Comment 5 — DR parameter table**: Added a table after the DR section listing domain randomization parameters: muscle inertia (alpha=0.7), max acceleration (a_max=0.15), observation noise (sigma ~ U(0.01, 0.08)), neural delay (0-3 frames / 0-375ms).



## v9 (archive: methodology_ieee_v9.docx)
- Replaced entire B. Dual-Stream Encoder section with user's rewritten version
- Expanded total loss to L_policy + c_v * L_value + c_e * L_entropy + lambda * L_aux
- Added self-supervised Markovian task explanation
- Added fusion layer equation

## v8 (archive: methodology_ieee_v7.docx)
- Fixed ZPD introduction flow (principle → formalization)
- Fixed MDP logic bridge (added "why RL" reasoning)
- Trimmed reward paragraph detail

## v7 (archive: methodology_ieee_v7.docx)
- Removed YOLO and MediaPipe from Hardware Platform
- Added signal path: camera→USB→computer→Ethernet→robot, RTDE interface
