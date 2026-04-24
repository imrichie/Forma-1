import tensorflow as tf
import matplotlib.pyplot as plt

from model.diffusion import get_sinusoidal_embedding

def sample_ddpm(model, num_samples, image_size, T,
                beta, alpha, alpha_bar, time_dim):
    """
    Full DDPM reverse process — 1000 denoising steps.
    Slower but follows the original paper exactly.
    """
    x = tf.random.normal((num_samples, image_size, image_size, 3))

    for t_val in reversed(range(T)):
        t_tensor = tf.ones((num_samples,), dtype=tf.int32) * t_val
        time_emb = get_sinusoidal_embedding(t_tensor, time_dim)

        z = tf.random.normal(tf.shape(x)) if t_val > 0 else tf.zeros_like(x)

        beta_t      = beta[t_val]
        alpha_t     = alpha[t_val]
        alpha_bar_t = alpha_bar[t_val]

        predicted_noise = model([x, time_emb], training=False)
        x = (1.0 / tf.sqrt(alpha_t)) * (
            x - (1.0 - alpha_t) / tf.sqrt(1.0 - alpha_bar_t) * predicted_noise
        )

        if t_val > 0:
            x = x + tf.sqrt(beta_t) * z

    return tf.clip_by_value((x + 1.0) / 2.0, 0.0, 1.0)


def sample_ddim(model, num_samples, image_size, T,
                alpha_bar, time_dim, num_steps=50):
    """
    DDIM reverse process — deterministic, ~20x faster than DDPM.
    Uses 50 evenly spaced steps instead of all 1000.
    Same model, no retraining needed.
    """
    x         = tf.random.normal((num_samples, image_size, image_size, 3))
    step_size = T // num_steps
    timesteps = list(range(0, T, step_size))[::-1]

    for i, t_val in enumerate(timesteps):
        t_tensor    = tf.ones((num_samples,), dtype=tf.int32) * t_val
        time_emb    = get_sinusoidal_embedding(t_tensor, time_dim)
        alpha_bar_t = alpha_bar[t_val]

        predicted_noise = model([x, time_emb], training=False)

        # estimate clean image from current noisy prediction
        x0_pred = (x - tf.sqrt(1.0 - alpha_bar_t) * predicted_noise) / tf.sqrt(alpha_bar_t)
        x0_pred = tf.clip_by_value(x0_pred, -1.0, 1.0)

        if i == len(timesteps) - 1:
            x = x0_pred
        else:
            t_prev         = timesteps[i + 1]
            alpha_bar_prev = alpha_bar[t_prev]
            x = tf.sqrt(alpha_bar_prev) * x0_pred + tf.sqrt(1.0 - alpha_bar_prev) * predicted_noise

    return tf.clip_by_value((x + 1.0) / 2.0, 0.0, 1.0)


def sample_with_steps(model, image_size, T,
                      beta, alpha, alpha_bar,
                      time_dim, num_snapshots=10):
    """
    Runs the full DDPM reverse process on a single image,
    capturing snapshots at regular intervals to visualize
    the denoising progression from noise to UI.
    """
    x                 = tf.random.normal((1, image_size, image_size, 3))
    snapshots         = []
    snapshot_interval = T // num_snapshots

    for t_val in reversed(range(T)):
        t_tensor    = tf.ones((1,), dtype=tf.int32) * t_val
        time_emb    = get_sinusoidal_embedding(t_tensor, time_dim)
        z           = tf.random.normal(tf.shape(x)) if t_val > 0 else tf.zeros_like(x)

        beta_t      = beta[t_val]
        alpha_t     = alpha[t_val]
        alpha_bar_t = alpha_bar[t_val]

        predicted_noise = model([x, time_emb], training=False)
        x = (1.0 / tf.sqrt(alpha_t)) * (
            x - (1.0 - alpha_t) / tf.sqrt(1.0 - alpha_bar_t) * predicted_noise
        )

        if t_val > 0:
            x = x + tf.sqrt(beta_t) * z

        if t_val % snapshot_interval == 0:
            snapshot = tf.clip_by_value((x + 1.0) / 2.0, 0.0, 1.0)
            snapshots.append((t_val, snapshot[0].numpy()))

    return snapshots


def plot_grid(images, title, grid_size=6):
    """Plots a square grid of generated images."""
    plt.figure(figsize=(12, 12))
    for i in range(grid_size * grid_size):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_denoising_steps(snapshots):
    """Plots the denoising progression snapshots side by side."""
    plt.figure(figsize=(len(snapshots) * 2, 3))
    for i, (t_val, snap) in enumerate(snapshots):
        plt.subplot(1, len(snapshots), i + 1)
        plt.imshow(snap)
        plt.axis('off')
        plt.title(f't={t_val}', fontsize=8)
    plt.suptitle('Forma-1 — Noise to UI')
    plt.tight_layout()
    plt.show()