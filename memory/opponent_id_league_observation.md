# Opponent ID League Training Observation

Date: 2026-06-17

Compared two league training runs:

- `logs/dual_iterative_0616_1612`: no opponent id
- `logs/dual_iterative_0616_2246`: with opponent id appended to robot observation

## Setup

The experiment tests whether multi-opponent robot training fails because of irreducible strategy conflict or because the robot observation/history does not provide enough opponent identity information.

The key intervention is adding a one-hot opponent id to the robot observation while keeping the same PFSP opponent pool structure.

## Main finding

Adding opponent id substantially stabilizes multi-opponent pool training. The evidence supports opponent identity aliasing / partial observability as the main cause of the no-id pool training failure, rather than the task itself being unlearnable due to strategy conflict.

## TensorBoard comparison

No opponent id, `dual_iterative_0616_1612`:

```text
iteration_1 robot:
eval/mean_reward tail5 ≈ -4.07
eval/mean_ep_length tail5 ≈ 78.17

iteration_2 robot:
eval/mean_reward tail5 ≈ -33.50
eval/mean_ep_length tail5 ≈ 30.65

iteration_3 robot:
eval/mean_reward tail5 ≈ -32.68
eval/mean_ep_length tail5 ≈ 30.72
```

With opponent id, `dual_iterative_0616_2246`:

```text
iteration_1 robot:
eval/mean_reward tail5 ≈ 1.05
eval/mean_ep_length tail5 ≈ 87.17

iteration_2 robot:
eval/mean_reward tail5 ≈ -5.10
eval/mean_ep_length tail5 ≈ 67.69

iteration_3 robot:
eval/mean_reward tail5 ≈ -2.65
eval/mean_ep_length tail5 ≈ 68.67
```

## Per-opponent episode evidence

In the no-id run, the robot still performs well against the scripted hand but fails badly against learned hands:

```text
No-id iteration 2:
scripted_hand: reward ≈ -14.27, ep_len ≈ 75.2
iteration_1 learned hand: reward ≈ -41.57, ep_len ≈ 17.2

No-id iteration 3:
scripted_hand: reward ≈ -11.41, ep_len ≈ 81.1
iteration_1 learned hand: reward ≈ -41.53, ep_len ≈ 17.8
iteration_2 learned hand: reward ≈ -41.42, ep_len ≈ 17.3
```

This shows the robot is not globally broken; it preserves a useful strategy for the scripted hand but cannot choose the right behavior for learned opponents when identity is hidden.

## Sampling was not the main cause

PFSP distributions were similar across runs:

```text
iteration_2: [0.2, 0.8]
iteration_3: [0.2, approximately 0.40, approximately 0.40]
```

Therefore the large convergence difference is not explained by a different opponent sampling distribution.

## Optimizer and auxiliary loss were not the main cause

The no-id run had reasonable value-function explained variance and low auxiliary prediction loss, but still failed in multi-opponent training. The with-id run could perform better even when auxiliary loss was not consistently lower. This suggests the improvement comes from resolving opponent identity ambiguity, not merely from better auxiliary dynamics prediction or PPO optimization.

## Interpretation for paper / analysis

The multi-opponent pool creates a mixed task. Without an explicit opponent id, similar geometric states may require different robot actions depending on which hand policy is active. The 16-frame LSTM history did not reliably disambiguate opponent identity. Adding one-hot identity removes this state aliasing and lets the shared policy condition its behavior on the current opponent.

Recommended phrasing:

```text
The non-ID league agent failed to improve after the opponent pool expanded, despite stable PPO optimization and similar PFSP sampling. Episode-level analysis showed that it retained reasonable performance against the scripted opponent but consistently failed against learned opponents. Adding a one-hot opponent identity substantially improved evaluation reward and episode length in later iterations, indicating that the original temporal history was insufficient to disambiguate opponent dynamics. Therefore, the observed league instability is primarily attributable to partial observability / opponent-identity aliasing rather than irreducible strategic conflict.
```
