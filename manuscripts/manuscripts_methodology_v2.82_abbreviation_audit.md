# Abbreviation audit for manuscripts_methodology_v2.82_reference_fixes.docx

## Required corrections

### 1. PPO is used without definition

First occurrence, Section II.B:

> The reward signal used for PPO policy learning [28] is defined as follows:

Recommended:

> The reward signal used for policy learning with proximal policy optimization (PPO) [28] is defined as follows:

PPO appears again in the objective description, league-training motivation, Algorithm 1, and the Supplementary Material pointer, so retaining the abbreviation is justified.

### 2. APF is not explicitly linked to its full term

The Introduction uses "artificial potential field methods," but later Section IV.B and Table II use APF without an explicit definition.

Recommended first-use revision:

> Rule-based reactive controllers, such as artificial potential field (APF) methods, can respond to the current hand-microrobot geometry...

Then Section IV.B can use:

> The comparison includes the reactive APF baseline...

### 3. YOLO is used only once and is not defined

Current deployment sentence:

> The perception pipeline combined YOLO-based microrobot detection [48]...

Because the paper now cites only the original YOLO paper and YOLO appears once, an abbreviation is unnecessary. Recommended:

> The perception pipeline combined microrobot detection based on the original You Only Look Once model [48]...

### 4. PFSP is defined but never reused

Current sentence:

> ...prioritized fictitious self-play (PFSP) sampling were not computational bottlenecks...

PFSP occurs only once. Recommended:

> ...prioritized fictitious self-play sampling were not computational bottlenecks...

Alternatively, PFSP would need to be used consistently in the following league-sampling paragraph, but removing the one-off abbreviation is cleaner.

## Consistency corrections

### 5. SPC full-name usage is inconsistent

The first definition is:

> stochastic scripted pursuit controller (SPC)

Later passages repeatedly use "scripted pursuit controller" without "stochastic," while other passages use SPC. After the first definition, use SPC consistently. Suggested revisions include:

- "a pool P that contains the SPC and learned hand policies"
- "a final pool that contains the SPC and ten learned hand policies"
- Algorithm input: "SPC policy π_H^SPC"
- "The SPC remains in the pool with a fixed sampling probability"

The baseline name also varies between "SPC baseline" and "SPC-only baseline." Use "SPC-only baseline" throughout.

### 6. TIZ alternates with the expanded metric name after definition

Time in Zone (TIZ) is correctly defined in Section IV.A, but later text returns to "Time in Zone" or "time within the ZPD." For a quantitative metric, consistent use of TIZ is preferable after definition.

Recommended examples:

- "Recurrent history increases TIZ and workspace coverage..."
- Fig. 4 caption: "higher values are better for TIZ and workspace coverage"
- Discussion: "TIZ increased from 0.161..."
- Discussion: "mean TIZ across learned hands increased..."
- Limitations: "recent TIZ"
- Conclusion: "improved both mean and minimum TIZ"

### 7. ZPD is unnecessarily expanded again in the Conclusion

The abstract correctly spells out Zone of Proximal Development because the abstract is self-contained. The main text then defines Zone of Proximal Development (ZPD) in the Introduction. The Conclusion can therefore use:

> Patient-calibrated ZPD settings...

## Abbreviations already handled correctly

- RL: defined at first abbreviated use in the Introduction.
- ZPD: defined at first abbreviated use in the Introduction; the abstract uses the full term without the abbreviation, which is appropriate.
- MLP: defined as multilayer perceptron before subsequent use.
- GRU: defined as gated recurrent unit before subsequent use.
- SPC: initially defined correctly; only later consistency needs revision.
- TIZ: initially defined correctly before table and figure use.
- H1: explained as the hand policy from the first hand-training phase before its table entries.
- UR10: a hardware model designation rather than an acronym requiring expansion.
- 2D: a standard technical abbreviation; no definition is required, although "planar positions" could replace this one-off use if desired.

## Related naming issue

The same baseline is called "single-policy baseline" in Section IV.A and "single-opponent baseline" in Section IV.B and Table II. This is not an acronym error, but one name should be used throughout. "Single-opponent baseline" is more precise because the distinction concerns the training opponent.
