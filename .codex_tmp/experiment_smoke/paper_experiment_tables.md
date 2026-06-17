# Paper Experiment Results

Generated: 2026-06-03 21:48:50
ZPD band: [4.0, 6.0] environment units.

Note: `human_proxy` is an automated mouse-like pursuit controller. Run an interactive human-mouse test separately before claiming human-in-the-loop results.

## Network Architecture Ablation

| Architecture | Reward | Episode Length | ZPD Coverage | Avg Distance | Too-close | Too-far | Aux MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP+LSTM | -17.33 +/- 21.13 | 3.67 +/- 2.31 | 0.00 +/- 0.00% | 7.47 +/- 5.35 | 33.33 +/- 57.74% | 66.67 +/- 57.74% | - |
| MLP+LSTM+Aux | -17.21 +/- 21.23 | 3.67 +/- 2.31 | 0.00 +/- 0.00% | 7.80 +/- 5.39 | 33.33 +/- 57.74% | 66.67 +/- 57.74% | 0.26 +/- 0.11 |
| MLP-Only | -17.08 +/- 20.84 | 3.67 +/- 2.31 | 0.00 +/- 0.00% | 7.64 +/- 5.37 | 33.33 +/- 57.74% | 66.67 +/- 57.74% | - |

## League Training / OOD Generalization

| Test Protocol | Method | ZPD Coverage | Avg Distance | Too-close | Too-far | Episode Length | Catch Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sluggish | League | 0.00 +/- 0.00% | 12.24 +/- 0.00 | 0.00 +/- 0.00% | 100.00 +/- 0.00% | 5.00 +/- 0.00 | 0.00 +/- 0.00% |
| sluggish | Scripted Only | 0.00 +/- 0.00% | 12.20 +/- 0.00 | 0.00 +/- 0.00% | 100.00 +/- 0.00% | 5.00 +/- 0.00 | 0.00 +/- 0.00% |
| sluggish | Single RL | 0.00 +/- 0.00% | 12.22 +/- 0.00 | 0.00 +/- 0.00% | 100.00 +/- 0.00% | 5.00 +/- 0.00 | 0.00 +/- 0.00% |
| spasm | League | 0.00 +/- 0.00% | 12.15 +/- 0.00 | 0.00 +/- 0.00% | 100.00 +/- 0.00% | 5.00 +/- 0.00 | 0.00 +/- 0.00% |
| spasm | Scripted Only | 0.00 +/- 0.00% | 12.11 +/- 0.00 | 0.00 +/- 0.00% | 100.00 +/- 0.00% | 5.00 +/- 0.00 | 0.00 +/- 0.00% |
| spasm | Single RL | 0.00 +/- 0.00% | 12.13 +/- 0.00 | 0.00 +/- 0.00% | 100.00 +/- 0.00% | 5.00 +/- 0.00 | 0.00 +/- 0.00% |
| delayed | League | 0.00 +/- 0.00% | 12.40 +/- 0.00 | 0.00 +/- 0.00% | 100.00 +/- 0.00% | 5.00 +/- 0.00 | 0.00 +/- 0.00% |
| delayed | Scripted Only | 0.00 +/- 0.00% | 12.37 +/- 0.00 | 0.00 +/- 0.00% | 100.00 +/- 0.00% | 5.00 +/- 0.00 | 0.00 +/- 0.00% |
| delayed | Single RL | 0.00 +/- 0.00% | 12.39 +/- 0.00 | 0.00 +/- 0.00% | 100.00 +/- 0.00% | 5.00 +/- 0.00 | 0.00 +/- 0.00% |
| noisy | League | 0.00 +/- 0.00% | 11.92 +/- 0.00 | 0.00 +/- 0.00% | 100.00 +/- 0.00% | 5.00 +/- 0.00 | 0.00 +/- 0.00% |
| noisy | Scripted Only | 0.00 +/- 0.00% | 11.89 +/- 0.00 | 0.00 +/- 0.00% | 100.00 +/- 0.00% | 5.00 +/- 0.00 | 0.00 +/- 0.00% |
| noisy | Single RL | 0.00 +/- 0.00% | 11.89 +/- 0.00 | 0.00 +/- 0.00% | 100.00 +/- 0.00% | 5.00 +/- 0.00 | 0.00 +/- 0.00% |
| unseen_rl | League | 0.00 +/- 0.00% | 12.09 +/- 0.00 | 0.00 +/- 0.00% | 100.00 +/- 0.00% | 5.00 +/- 0.00 | 0.00 +/- 0.00% |
| unseen_rl | Scripted Only | 0.00 +/- 0.00% | 12.08 +/- 0.00 | 0.00 +/- 0.00% | 100.00 +/- 0.00% | 5.00 +/- 0.00 | 0.00 +/- 0.00% |
| unseen_rl | Single RL | 0.00 +/- 0.00% | 12.09 +/- 0.00 | 0.00 +/- 0.00% | 100.00 +/- 0.00% | 5.00 +/- 0.00 | 0.00 +/- 0.00% |
| human_proxy | League | 0.00 +/- 0.00% | 11.77 +/- 0.00 | 0.00 +/- 0.00% | 100.00 +/- 0.00% | 5.00 +/- 0.00 | 0.00 +/- 0.00% |
| human_proxy | Scripted Only | 0.00 +/- 0.00% | 11.74 +/- 0.00 | 0.00 +/- 0.00% | 100.00 +/- 0.00% | 5.00 +/- 0.00 | 0.00 +/- 0.00% |
| human_proxy | Single RL | 0.00 +/- 0.00% | 11.75 +/- 0.00 | 0.00 +/- 0.00% | 100.00 +/- 0.00% | 5.00 +/- 0.00 | 0.00 +/- 0.00% |

## Domain Randomization Ablation Proxy

Existing checkpoints do not isolate CMD-DR factors cleanly, so this table is a checkpoint proxy: Baseline A as scripted/no-DR, Baseline B as single-RL intent exposure, and League as the full heterogeneous training setting.

| Training Setting | Sluggish ZPD | Spasm ZPD | Delayed ZPD | Noisy ZPD | Unseen RL ZPD |
| --- | ---: | ---: | ---: | ---: | ---: |
| No DR / Scripted | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Intent Exposure / Single RL | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Full CMD-DR + League | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

## League Mechanism / Checkpoint Progression Proxy

| League Variant | ZPD Coverage | Avg Distance | OOD Mean Score | Worst-case ZPD |
| --- | ---: | ---: | ---: | ---: |
| Full League (iter 10) | 24.58% | 6.67 | 24.58% | 23.33% |
| No Script Anchor Proxy (iter 6) | 30.36% | 6.67 | 30.36% | 28.57% |
| Prioritized Pool Proxy (iter 3) | 33.55% | 6.92 | 33.55% | 30.36% |
| Uniform Pool Proxy (iter 1) | 22.77% | 6.71 | 22.77% | 17.86% |
