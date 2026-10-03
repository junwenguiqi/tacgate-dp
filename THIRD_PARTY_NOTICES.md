# TacGate-DP: third-party code and attribution

TacGate-DP is the project name; `tactile_gate_policy` remains the Python import
namespace. This document identifies reused code and dependencies. Research
inspiration and method comparisons are documented separately in
[Related Work](docs/RELATED_WORK.md).

The `src/tactile_gate_policy/vendor` directory includes a minimal, adapted
subset of [Diffusion Policy](https://github.com/real-stanford/diffusion_policy),
originally distributed under the MIT license:

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab.

The original license is distributed as `vendor/LICENSE.diffusion_policy`.
The vendored action U-Net and convolution / positional embedding components
retain the original architecture. The vision encoder follows DP's
`get_resnet` / `MultiImageObsEncoder` ResNet18, GroupNorm, crop and ImageNet
normalization pipeline while exposing camera features separately for gating.
File headers identify the original source files and adaptations.

The package also depends on PyTorch, Torchvision, Diffusers, Einops, NumPy,
Pandas, PyArrow, Zarr, Pillow, and PyYAML. Those dependencies are installed
separately and remain governed by their own licenses. No pretrained weights,
datasets or experiment results are bundled.

## Implementation and research attribution

The observation-only time/modality Router, feature gate, separate fusion
Transformer, taxel-index tactile CNN, and release training/data/evaluation
integration are implemented in this repository. Their implementation does not
establish that the underlying mechanisms are new research contributions.

Policy Consensus and Multimodal Diffusion Forcing informed the initial project
design. FoAR, Touch in the Wild, Dream-Tac, CAAT, and OmniVTA are related work
identified during the public-source review. They are comparisons and research
references, not additional vendored code dependencies. No claim is made that
gating, learned mask embeddings, attention fusion, or tactile-conditioned
diffusion were first introduced by TacGate-DP.
