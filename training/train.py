import os
import numpy as np
import tensorflow as tf

from model.diffusion import get_sinusoidal_embedding

def train_step(model, optimizer, x, alpha_bar, T, time_dim):
    """
    Single training step.
    Samples a random timestep, corrupts the image, predicts the noise,
    and updates the model weights via MSE loss.
    """
    batch_size  = tf.shape(x)[0]
    t           = tf.random.uniform((batch_size,), 0, T, dtype=tf.int32)
    alpha_bar_t = tf.gather(alpha_bar, t)
    alpha_bar_t = tf.reshape(alpha_bar_t, (-1, 1, 1, 1))

    noise    = tf.random.normal(tf.shape(x))
    x_t      = tf.sqrt(alpha_bar_t) * x + tf.sqrt(1.0 - alpha_bar_t) * noise
    time_emb = get_sinusoidal_embedding(t, time_dim)

    with tf.GradientTape() as tape:
        noise_predicted = model([x_t, time_emb], training=True)
        loss = tf.reduce_mean((noise - noise_predicted) ** 2)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def run_training(model, optimizer, dataset, alpha_bar,
                 T, time_dim, epochs, checkpoint_dir,
                 start_epoch=0, all_losses=None):
    """
    Full training loop with checkpointing every 10 epochs.
    Resumes automatically if a checkpoint exists.
    Tracks average loss per epoch across all batches (not just the last).
    """
    if all_losses is None:
        all_losses = []

    weights_path = os.path.join(checkpoint_dir, 'forma1_weights.weights.h5')
    history_path = os.path.join(checkpoint_dir, 'forma1_history.npy')

    for epoch in range(start_epoch, epochs):
        epoch_losses = []

        for batch in dataset:
            loss = train_step(model, optimizer, batch, alpha_bar, T, time_dim)
            epoch_losses.append(float(loss.numpy()))

        # average loss across all batches in this epoch (fix #1)
        epoch_avg_loss = np.mean(epoch_losses)
        all_losses.append(epoch_avg_loss)
        print(f'Epoch: {epoch + 1}/{epochs} | Loss: {round(epoch_avg_loss, 4)}')

        # checkpoint every 10 epochs — protects against Colab disconnects
        if (epoch + 1) % 10 == 0:
            model.save_weights(weights_path)
            np.save(history_path, np.array(all_losses))
            print(f'  └─ checkpoint saved at epoch {epoch + 1}')

    # final save
    model.save_weights(weights_path)
    np.save(history_path, np.array(all_losses))
    print('Training complete')

    return all_losses