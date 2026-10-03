# TacGate-DP method

**TacGate-DP: Time–Modality Gated Tactile Conditioning for Diffusion Policies**

TacGate-DP controls how historical tactile features enter an action diffusion policy. Independent left and right tactile CNNs produce observation tokens. An observation-conditioned router assigns a gate to every historical time–modality pair. Gated tactile, visual, and proprioceptive tokens are fused into a global condition, which modulates the action denoiser through FiLM.

This document describes the released implementation and its default configuration. The Python import namespace remains `tactile_gate_policy`; the installable distribution is `tacgate-dp`. See the [README](../README.md) for installation and commands, the [architecture PDF](../figures/model_architecture.pdf) for the method figure, and [related work](RELATED_WORK.md) for the contribution boundaries.

## 1. Observations and notation

The observation window contains only historical measurements. Let $B$ be the batch size, $T_o=4$ the observation horizon, $M=5$ the number of modalities, and $D=256$ the token width. Modalities have the fixed order

$$
\mathcal M=(\text{global RGB},\text{wrist RGB},\text{left tactile},\text{right tactile},\text{proprioception}).
$$

Each encoder produces one token for each observation time. The resulting tensor is

$$
Z=\{z_{b,t,m}\}\in\mathbb R^{B\times T_o\times M\times D}.
$$

There are 20 observation tokens per sample. The action horizon is $H=8$, with $d_a=8$ coordinates per action: position $(x,y,z)$, quaternion $(q_x,q_y,q_z,q_w)$, and gripper command. The rollout interface executes the first four predicted actions and then replans. The default control interval is 0.3 s.

| Quantity | Default shape | Meaning |
|---|---|---|
| Each RGB stream | $B\times4\times3\times96\times96$ after resizing | Global or wrist RGB; cropped to $84\times84$ before the backbone |
| Each tactile stream | $B\times4\times231$ | Raw distribution values; interpreted as 77 interleaved three-channel taxels by default |
| Proprioception | $B\times4\times22$ | EE pose 7, joint positions 7, joint velocities 7, gripper position 1 |
| Encoded observations $Z$ | $B\times4\times5\times256$ | One token per time and modality |
| Gates $G$ | $B\times4\times5$ | Independent scalar gates in $[0,1]$ |
| Gated observations $Z_g$ | $B\times4\times5\times256$ | All token positions retained |
| Fusion sequence | $B\times20\times256$ | Flattened time and modality axes |
| Global condition $c$ | $B\times256$ | Shared across an action chunk's reverse diffusion steps |
| Noisy action $A_k$, noise estimate, full action prediction | $B\times8\times8$ | Action sequence at diffusion level $k$ |
| Execution prefix | $B\times4\times8$ | First four actions returned for execution |

The dataset uses a frame stride of three by default. A 10 Hz source therefore yields the configured 0.3 s control interval. The loader checks timestamps rather than assuming that a chosen stride always matches this interval.

## 2. Modality encoders

The implementations are in [encoders.py](../src/tactile_gate_policy/encoders.py), with defaults in [config.py](../src/tactile_gate_policy/config.py).

### Visual encoders

The global and wrist cameras use two independently parameterized DP-style ResNet18 encoders by default. The ResNet classification head is removed and BatchNorm is replaced with GroupNorm. Each 512-dimensional feature is projected by `Linear(512, 256) → LayerNorm`.

Images are resized to $96\times96$. Training uses an independently sampled $84\times84$ crop for each frame; evaluation uses a center crop. ImageNet input normalization is enabled. The default `vision_weights: null` initializes the backbone without downloading pretrained weights. Backbone sharing and explicit torchvision pretrained weights are configurable, but are not the defaults.

### Tactile encoders

The left and right tactile sensors use independent three-layer Conv1d encoders. Raw tactile distribution values are divided by 255. With `tactile_dim=231` and `tactile_channels=3`, each vector is reshaped into $77\times3$, then transposed to three channels over 77 taxel indices.

Each layer applies a kernel-5, stride-2, padding-2 convolution followed by GroupNorm and GELU. The channel widths are $32\rightarrow64\rightarrow128$. Adaptive average pooling over the remaining index axis gives a 128-dimensional vector; dropout, `Linear(128, 256)`, and LayerNorm produce the token.

This is a convolution over the sensor's index sequence. The implementation does not establish that consecutive indices are physical neighbors and does not invent a two-dimensional tactile map. The interleaved three-channel layout must agree with the actual data format. A one-channel configuration is available for sources that do not have this layout.

### Proprioceptive encoder

The 22-dimensional standardized state is encoded by `Linear(22, 256) → GELU → Dropout → Linear(256, 256) → LayerNorm`.

## 3. Observation-conditioned time–modality routing

The router in [model.py](../src/tactile_gate_policy/model.py) receives $Z$, adds learned time and modality embeddings, and applies a two-layer Transformer encoder with four attention heads per layer. Each feed-forward block has width $4D$. A LayerNorm and scalar linear head followed by Sigmoid produce

$$
G_{b,t,m}=\sigma\!\left(h_R\!\left[\operatorname{Transformer}_R
\left(\operatorname{flatten}_{t,m}(Z+E^R_t+E^R_m)\right)\right]_{b,t,m}\right).
$$

Every gate is independent in its output parameterization: there is no softmax constraint requiring the five modalities to share a fixed unit sum. The Transformer nevertheless allows a gate to depend on all tokens in the historical observation window. In particular, left and right tactile participation can vary with time and with visual and robot-state context.

The router does not receive the noisy action $A_k$ or diffusion index $k$. Its intermediate Transformer features generate gates; the observation features passed to the fusion path are the original encoder outputs $Z$, after the gating operation below.

## 4. Feature gating and fusion

For each modality, the model learns a mask vector $m_m\in\mathbb R^D$, initialized to zero and shared across observation times. The gated token is

$$
z^g_{b,t,m}=G_{b,t,m}z_{b,t,m}+(1-G_{b,t,m})m_m.
$$

One gate scales all channels of a token. A gate near zero replaces most of that token's content with a learned modality-specific representation. A gate near one preserves most of the encoded observation. All 20 positions remain in the sequence, including positions with low gates.

The fusion module is a second, independently parameterized two-layer, four-head Transformer. It adds its own time and modality embeddings to the gated tokens:

$$
U=\operatorname{Transformer}_F\!\left(
\operatorname{flatten}_{t,m}(Z_g+E^F_t+E^F_m)\right)
\in\mathbb R^{B\times20\times256}.
$$

Self-attention permits interaction across all observed times and modalities. The fused sequence is then averaged and projected:

$$
c=P\!\left(\frac{1}{T_oM}\sum_{i=1}^{T_oM}U_i\right),
\qquad
P=\operatorname{LN}\circ\operatorname{GELU}\circ\operatorname{Linear}\circ\operatorname{LN}.
$$

Here the composition is applied right to left, matching `LayerNorm → Linear → GELU → LayerNorm` in code. The output is $c\in\mathbb R^{B\times256}$. This average operates on tokens after their attention interactions; it is not a gate-weighted average of the original encoder features.

## 5. How tactile information enters action diffusion

The tactile pathway has three explicit stages:

1. **Representation:** the left and right CNNs place tactile features in the observation sequence.
2. **Participation control:** $G_{t,\mathrm{left}}$ and $G_{t,\mathrm{right}}$ control each historical tactile token's interpolation with its learned mask.
3. **Denoiser conditioning:** fusion forms $c$, and the DP action U-Net uses $c$ to modulate intermediate features through FiLM.

The action noise predictor is

$$
\widehat\epsilon=\epsilon_\theta(A_k,k,c).
$$

The local [ConditionalUnet1D implementation](../src/tactile_gate_policy/vendor/diffusion_policy/conditional_unet1d.py) concatenates the 128-dimensional diffusion-step embedding $e(k)$ with $c$. Conditional residual blocks map this 384-dimensional vector to feature-wise scales and biases. With scale prediction enabled, the implementation applies

$$
\widetilde h_\ell=\gamma_\ell([e(k),c])\odot h_\ell+\beta_\ell([e(k),c]).
$$

The default U-Net widths are 64 and 128, with encoder–decoder skip connections. FiLM parameters depend on both $k$ and $c$, so their values can change across diffusion steps even though $c$ is fixed for that action chunk.

Tactile information enters through a **mixed visual–tactile–proprioceptive condition**. The release does not have a separate tactile-only adapter attached to an independently pretrained, frozen DP checkpoint. Both the conditioning modules and action denoiser participate in the released training procedure. A gate value measures the learned interpolation strength at an input token; it is not a calibrated causal importance score.

## 6. Two-stage training

The training objective and gate sampler are in [policy.py](../src/tactile_gate_policy/policy.py); parameter freezing is controlled by [train.py](../src/tactile_gate_policy/train.py).

For normalized action labels $A_0$, a uniformly sampled diffusion level $k\in\{0,\ldots,99\}$, and Gaussian noise $\epsilon\sim\mathcal N(0,I)$, training constructs

$$
A_k=\sqrt{\bar\alpha_k}A_0+\sqrt{1-\bar\alpha_k}\epsilon,
\qquad
\mathcal L_\epsilon=\mathbb E\left[\|\epsilon_\theta(A_k,k,c)-\epsilon\|_2^2\right].
$$

The implemented MSE averages over batch, action horizon, and action coordinates. The DDPM uses a cosine beta schedule, epsilon prediction, fixed-small variance, and no clipping of standardized action samples.

### Stage A: random-gate action pretraining

The [Stage A configuration](../configs/stage_a.yaml) bypasses and freezes the router. For each sample, one keep probability $p_b\sim\mathcal U(0.35,0.95)$ is sampled; binary gates are drawn independently at every time–modality position conditional on $p_b$. If an entire time row would be masked, one uniformly selected modality is restored.

Encoders, mask vectors, the fusion module, and the action U-Net are trainable by default. The optimized loss is only $\mathcal L_\epsilon$. This exposes the action model to varying observation subsets; it is **action diffusion pretraining, not tactile reconstruction pretraining**. Robustness benefits remain an experimental question.

### Stage B: learned-gate optimization

The [Stage B configuration](../configs/stage_b.yaml) initializes from a checkpoint, enables the observation router, and freezes the encoders by default. The router, mask vectors, fusion module, and action U-Net are trainable. The default objective is

$$
\mathcal L_B=\mathcal L_\epsilon
+0.1\left(\operatorname{mean}(G)-0.65\right)^2
+0.02\operatorname{mean}_{b,t,m}|G_{b,t+1,m}-G_{b,t,m}|.
$$

The gate-mean term is computed over all batch, time, and modality entries. It encourages a mean interpolation level rather than enforcing a per-sample token count. The temporal term applies within the current historical window; there is no separate cross-rollout-window continuity loss.

An optional left–right tactile symmetry term is

$$
\mathcal L_{\mathrm{sym}}=\operatorname{mean}_{b,t}
|G_{b,t,\mathrm{left}}-G_{b,t,\mathrm{right}}|.
$$

Its default coefficient is zero, allowing the two sensors to have different gate strengths. No contact labels or externally specified gate targets supervise the router.

## 7. Normalization and receding-horizon inference

The [numeric normalizer](../src/tactile_gate_policy/normalizer.py) fits state and action mean/standard deviation using only the training split. Tactile distribution values use division by 255. RGB input is either `uint8` or floating point in $[0,1]$, followed by the visual preprocessing above. Normalization buffers are saved with checkpoints and reused for inference.

For each action chunk, [predict_action](../src/tactile_gate_policy/policy.py) performs the following sequence:

1. Encode the current four-frame synchronized historical window.
2. Compute learned gates $G$ and fused condition $c$ once.
3. Draw a normalized $8\times8$ action-noise sequence and run the DDPM reverse process, reusing $c$ at each of the default 100 denoising steps.
4. Denormalize the predicted actions and normalize their quaternion components to unit length. A near-zero quaternion is replaced with the identity quaternion in `xyzw` order.
5. Return the full eight-step prediction and its four-step execution prefix.

The [rollout interface](../src/tactile_gate_policy/rollout.py) obtains an observation after each executed action, updates the historical buffer, and replans after the execution prefix unless the episode terminates. It does not use future ground-truth actions to form the condition. The release predicts actions only; future RGB or tactile signals are not diffusion targets.

## 8. Scope of the released evidence

The retained sequence length is always 20 under the default architecture. Gating changes feature content and does not remove tokens or establish inference compute savings. The release also does not implement action-conditioned routing, observation diffusion noise levels, a mixture of action denoisers, or a dedicated tactile residual adapter.

The [validation report](../VALIDATION.md) covers software behavior, CPU integration, checkpoint restoration, and a simulated adapter smoke test. It does not establish physical-robot success rates, tactile gains, or robustness to sensor corruption. The [related-work and ablation plan](RELATED_WORK.md) explains which comparisons are needed before claiming a validated methodological contribution.
