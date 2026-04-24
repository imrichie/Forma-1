import tensorflow as tf

def get_sinusoidal_embedding(t, dim):
    """
    Sinusoidal positional encoding for timestep t.
    Gives the model a rich continuous sense of where it is
    in the noising process — much better than a raw scalar.

    Args:
        t   — batch of integer timesteps, shape (batch_size,)
        dim — embedding dimension (must be even)

    Returns:
        embedding of shape (batch_size, dim)
    """
    half_dim   = dim // 2
    frequencies = tf.math.log(10000.0) / tf.cast(half_dim - 1, tf.float32)
    frequencies = tf.exp(tf.cast(tf.range(half_dim), tf.float32) * -frequencies)

    t_float = tf.cast(t, tf.float32)
    angles  = t_float[:, None] * frequencies[None, :]

    embedding = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)
    return embedding   # shape: (batch_size, dim)