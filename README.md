# Forma-1

`Forma-1` is a diffusion model trained on 36,536 mobile UI screenshots from the [RICO](https://interactionmining.org/rico) dataset. Give it random noise and it will denoise it into something that looks like a mobile app screen.

Built as part of DiffuseUI — a project exploring generative AI applied to interface design.

> 📸 *Generated samples coming after training completes*

---

## Details

| | |
|---|---|
| **Architecture** | U-Net (4-level encoder/decoder) |
| **Framework** | TensorFlow / Keras |
| **Image Size** | 128×128 |
| **Timesteps** | 1000 |
| **Noise Schedule** | Linear (β: 1e-4 → 0.02) |
| **Epochs** | 200 |
| **Batch Size** | 32 |
| **Learning Rate** | 1e-4 |
| **Loss** | MSE (noise prediction) |
| **Precision** | Mixed (float16 / float32) |
| **Sampler** | DDPM (1000 steps) + DDIM (50 steps) |
| **Hardware** | NVIDIA H100 via Google Colab Pro |

---

## Architecture

The U-Net denoiser takes two inputs at every step — the noisy image and a sinusoidal embedding of the current timestep — and predicts the noise that was added at that step.

```
Input: noisy image (128×128×3) + sinusoidal timestep embedding
       │
Encoder
  ├── Conv Block  64ch  →  128×128
  ├── Conv Block 128ch  →   64×64
  ├── Conv Block 256ch  →   32×32
  └── Conv Block 512ch  →   16×16
       │
Bottleneck
  └── Conv → Self-Attention → Conv   (8×8)
       │
Decoder  (skip connections from encoder at every level)
  ├── Upsample + Conv Block 256ch + Self-Attention  →  16×16
  ├── Upsample + Conv Block 256ch                   →  32×32
  ├── Upsample + Conv Block 128ch                   →  64×64
  └── Upsample + Conv Block  64ch                   → 128×128
       │
Output: predicted noise (128×128×3)
```

Key design choices:
- **GroupNorm over BatchNorm** — different timesteps produce different activation distributions, GroupNorm handles that more stably
- **Swish activations** — smoother gradients than ReLU for continuous noise prediction
- **Self-attention at 8×8 and 16×16** — captures long-range UI layout relationships across the full screen
- **Residual connections in every conv block** — preserves spatial detail through deep layers
- **Time embedding injected at every conv block** — the model always knows which noise level it is operating at

---

## How It Works

Standard DDPM setup. The forward process adds Gaussian noise to real UI screenshots across 1000 timesteps until they are pure static. The U-Net learns to predict that noise at each step. At generation time you start from pure Gaussian noise and run the reverse process — denoising 1000 times until a new UI screen comes out the other end.

Two samplers are included:

- **DDPM** — full 1000-step reverse process, follows the original paper exactly
- **DDIM** — deterministic 50-step sampler, ~20× faster generation using the same trained weights, no retraining needed

---

## Training Data

Trained on the [RICO](https://interactionmining.org/rico) dataset — 36,536 UI screenshots across 27 Android app categories including productivity, social, shopping, and entertainment.

**Preprocessing pipeline:**
- Center-cropped from 1440×2560 portrait to 1440×1440 square to preserve aspect ratio
- Resized to 128×128
- Normalized from [0, 255] to [−1, 1] for stable training
- Streamed via `tf.data` with parallel decoding and prefetching — nothing loaded into RAM

---

## Project Structure

```
Forma-1/
├── data/
│   └── preprocess.py          # RICO data loading + tf.data pipeline
├── model/
│   ├── noise_schedule.py      # Linear β schedule — beta, alpha, alpha_bar
│   ├── diffusion.py           # Sinusoidal timestep embedding
│   └── unet.py                # Full U-Net architecture
├── training/
│   └── train.py               # Train step + training loop with checkpointing
├── sampling/
│   └── sample.py              # DDPM sampler, DDIM sampler, denoising visualizer
├── forma-1.ipynb              # Colab orchestrator — imports from all modules above
└── requirements.txt
```

---

## Limitations

- **128×128 resolution** — outputs show clear UI structure but are not photorealistic
- **Unconditional** — no control over what category of UI gets generated
- **Android only** — trained exclusively on Android screenshots from the RICO dataset
- **Text rendering** — diffusion models struggle to generate legible text at this resolution

---

## Compared to Production Models

For perspective on scale:

| | Forma-1 | Stable Diffusion 1.4 |
|---|---|---|
| **Dataset** | 36,536 images | 5 billion images |
| **Resolution** | 128×128 | 512×512 |
| **Parameters** | ~50–80M | ~860M |
| **Training hardware** | 1× H100 | 256× A100 |
| **Training cost** | $0 | ~$600,000 |

The architecture and math are identical — the difference is data and compute, not theory.

---

## About

Built by Ricardo Flores as part of DiffuseUI.

[GitHub](https://github.com/imrichie) · [DiffuseUI](#)