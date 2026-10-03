# Related work and contribution scope

**Public-source audit date: 2026-10-02.** This document records the relevant repositories and papers examined for TacGate-DP. Repository availability and contents can change after that date. A finite public search cannot guarantee priority or rule out an equivalent unpublished or unindexed method.

TacGate-DP is an observation-conditioned time–modality feature-gating implementation with one action diffusion U-Net. Its tactile features enter the denoiser through a mixed visual–tactile–proprioceptive condition and FiLM. The precise released design is documented in [METHOD.md](METHOD.md).

## Close mechanisms and distinctions

| Work and primary sources | Shared mechanism | Difference from the release |
|---|---|---|
| **FoAR** — [official policy implementation](https://github.com/Alan-Heoooh/FoAR/blob/main/policy/policy.py#L70-L80), [repository](https://github.com/Alan-Heoooh/FoAR) | Soft interpolation between a force feature and a learnable embedding before action diffusion | FoAR uses a contact probability to control the force branch. TacGate-DP produces a separate observation-conditioned gate for each historical time and each of five modalities, followed by a separate fusion Transformer. |
| **Policy Consensus** — [paper](https://arxiv.org/abs/2509.23468), [official repository](https://github.com/policyconsensus/policyconsensus) | Adaptive routing of sensory contributions in a diffusion policy | Policy Consensus combines noise predictions from diffusion experts. TacGate-DP gates observation features and feeds one fused condition to one action U-Net. |
| **Touch in the Wild** — [paper](https://arxiv.org/abs/2507.15062), [official repository](https://github.com/XinyueZhuXY/touch_in_the_wild) | Tactile CNN features, learned masks, attention fusion, staged learning, and a diffusion policy | Its masked tactile-patch reconstruction pretraining differs from TacGate-DP's random binary observation gates and action-noise prediction objective in Stage A. |
| **Dream-Tac** — [paper](https://arxiv.org/abs/2606.08737), [official repository](https://github.com/LYFCLOUDFAN/Dream-Tac) | Time-varying tactile gating and attention-based multimodal diffusion | Its tactile frame-difference gate affects attention logits, with joint visual, tactile, and action generation. TacGate-DP learns observation-token gates, interpolates feature content, and diffuses actions only. |
| **CAAT** — [paper, including Appendix C](https://arxiv.org/html/2608.01102v1), [official repository](https://github.com/mrJiangjm/caat-code) | The paper includes a baseline with learnable modality scalars before joint self-attention | Those scalar weights establish a relevant feature-weighting and attention-fusion precedent. TacGate-DP uses historical-observation-conditioned time–modality gates and learned masks. The repository was explicitly a placeholder at the audit date, preventing a full implementation comparison. |
| **OmniVTA** — [paper](https://arxiv.org/abs/2603.19201), [official repository](https://github.com/MrSecant/OmniVTA) | Contact-conditioned channel gating with an action diffusion policy | TacGate-DP applies one scalar gate per time–modality token, without contact-label supervision. The linked repository contained only a license at the audit date, so the paper supports the mechanism comparison, not a full public-code comparison. |
| **Multimodal Diffusion Forcing (MDF)** — [paper](https://arxiv.org/abs/2511.04812), [official project page](https://unified-df.github.io/) | A time–modality matrix controls contributions in a multimodal diffusion setting | MDF assigns diffusion/noise levels and reconstructs modalities jointly. TacGate-DP's matrix contains feature-interpolation gates, with one action-only DDPM. The audit did not identify a released code repository through the official project page. |

The strongest direct overlap is FoAR's soft feature/mask interpolation. The expression

$$
Z_g=G\odot Z+(1-G)\odot m
$$

should therefore not be presented as a newly invented gating formula. Tactile CNNs, attention fusion, masked-modality training, staged learning, and FiLM conditioning likewise have prior uses. Differences in module placement or tensor shape can define a particular implementation without, by themselves, establishing a scientific advance.

## What belongs to this release

The current implementation combines the following choices:

- **Historical time–modality participation:** an observation-only Transformer predicts independent Sigmoid gates for all five modalities at every historical time, including separate left and right tactile gates.
- **Two distinct attention modules:** router features generate gates; independently parameterized attention then fuses the gated original observation features into one global action condition.
- **A specific training procedure:** random binary observation gates train the action-noise objective in Stage A; learned gates, a gate-mean target, and within-window temporal smoothing train the same action model in Stage B.
- **A reproducible FR3/Paxini software pipeline:** real image and tactile loading, training-only numeric statistics, split validation, checkpoint transfer and restoration, offline evaluation, and a robot adapter interface.

These are concrete design and implementation contributions. Their effectiveness relative to existing alternatives remains to be measured. No physical-robot results, benchmark improvements, gate interpretability guarantees, or publication-priority claims follow from the existing software checks.

A suitable current repository description is:

> TacGate-DP implements observation-conditioned time–modality gating for tactile conditioning of an action diffusion policy, with a two-stage training pipeline and reproducible data, checkpoint, evaluation, and rollout interfaces.

The model name expands to **Time–Modality Gated Tactile Conditioning for Diffusion Policies**. “Tactile conditioning” refers to the route from tactile CNN features through the fused global condition to FiLM in the action U-Net. It does not imply an independent tactile-only adapter or a frozen pretrained DP backbone.

## Controlled comparisons needed for a method claim

Keep the data split, encoders, fusion width/depth, U-Net, action horizon, training budget, and evaluation protocol consistent when comparing gates. Use multiple training seeds and report uncertainty for task-level results.

| Comparison | Question it addresses |
|---|---|
| All gates fixed to one, using the same encoders, fusion Transformer, and action U-Net | Does adaptive gating improve on otherwise identical multimodal conditioning? |
| Observation-conditioned modality-only gates $G_{b,m}$, broadcast over time | Does per-history-time resolution help beyond per-modality weighting? |
| A single tactile participation scalar, using the same action backbone | Does separate left/right and time-resolved tactile control help beyond a FoAR-style scalar design? |
| Remove random-gate Stage A, while matching total optimization steps | Does the two-stage schedule help beyond additional training compute? |
| Replace learned mask vectors with fixed zero vectors | Does the missing-modality representation contribute beyond feature scaling? |
| Remove the gate-mean term and temporal smoothness term separately, then together | Which regularizers affect performance and gate behavior? |
| Remove left tactile, right tactile, or both tactile inputs | Does the task benefit from tactile measurements, and is that benefit sensor-side-dependent? |
| Fixed, documented visual occlusion; tactile noise/dropout; and timestamp perturbations | Does gating improve behavior under specific sensor corruptions or misalignment? |

For a scalar-gate baseline, state exactly how the scalar is predicted and which streams it controls. A baseline inspired by another work is not an exact reproduction unless its released architecture, training procedure, and task protocol are reproduced.

The all-one baseline should receive its own matched training run. Setting gates to one only at evaluation changes the learned input distribution and answers a different intervention question. The same distinction applies to tactile removal: include matched training baselines when claiming modality value, and report evaluation-only interventions separately.

Action error, gate heatmaps, and gate averages help diagnose a model, but do not replace task success rates. A low tactile gate is not proof that touch is unnecessary: tactile observations can influence the router's other gates before feature interpolation, and learned representations and masks can interact with fusion. Claims about causal tactile use require carefully specified interventions.

## Code reuse, research references, and licensing

The code reuse documented in this release is the original **Diffusion Policy** implementation: the local action U-Net components and the adapted DP-style visual encoding path. Its attribution and MIT license are retained in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and the vendored license files. See the [official DP repository](https://github.com/real-stanford/diffusion_policy) and [project page](https://diffusion-policy.cs.columbia.edu/) for the source method.

The works listed above are research comparisons and potential experimental baselines. Listing a paper or repository as related work does not assert that its code was copied into this release. Research citations belong in scientific documentation; license notices should describe actual reused or redistributed material. Any future code adoption must be recorded with its applicable license and attribution.

The checked-in [validation report](../VALIDATION.md) documents engineering verification rather than research validation. CPU tests and an in-memory rollout smoke test establish that selected software paths execute and preserve specified invariants; they do not establish novelty or robot manipulation performance.
