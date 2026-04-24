import tensorflow as tf
from tensorflow.keras import layers

def conv_block(x, time_emb, filters):
    """
    ResNet-style block with time embedding injection and GroupNorm.
    GroupNorm is used instead of BatchNorm because different timesteps
    create different activation distributions — BatchNorm struggles with that.
    """
    shortcut = x

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.GroupNormalization(groups=8)(x)
    x = layers.Activation('swish')(x)

    # inject time embedding so the model knows its current noise level
    t = layers.Dense(filters)(time_emb)
    t = layers.Activation('swish')(t)
    x = x + t[:, None, None, :]

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.GroupNormalization(groups=8)(x)
    x = layers.Activation('swish')(x)

    # project shortcut to match channels if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)

    return layers.add([x, shortcut])


def attention_block(x):
    """
    Self-attention over spatial positions.
    Helps the model learn long-range layout relationships
    across the UI (e.g. nav bar relative to bottom tab bar).
    """
    h, w, c = x.shape[1], x.shape[2], x.shape[3]

    x_norm = layers.LayerNormalization()(x)
    x_seq  = layers.Reshape((h * w, c))(x_norm)

    attn = layers.MultiHeadAttention(num_heads=8, key_dim=c // 8)(x_seq, x_seq)
    attn = layers.Reshape((h, w, c))(attn)

    return layers.add([x, attn])


def get_unet(image_size=128, time_dim=256):
    """
    U-Net denoiser for DDPM.
    Takes a noisy image and a sinusoidal time embedding,
    predicts the noise that was added at that timestep.

    Architecture:
        Encoder : 128 → 64 → 32 → 16 → 8  (channels: 64, 128, 256, 512)
        Bottleneck : conv → attention → conv at 8x8
        Decoder : 8 → 16 → 32 → 64 → 128  (attention added at 16x16)
    """
    x_input = layers.Input(shape=(image_size, image_size, 3))
    t_input = layers.Input(shape=(time_dim,))

    # project time embedding through small MLP for richer representation
    t = layers.Dense(time_dim * 4)(t_input)
    t = layers.Activation('swish')(t)
    t = layers.Dense(time_dim * 4)(t)

    # encoder
    c1 = conv_block(x_input, t, 64)
    p1 = layers.Conv2D(64, 3, strides=2, padding='same')(c1)     # 128 → 64

    c2 = conv_block(p1, t, 128)
    p2 = layers.Conv2D(128, 3, strides=2, padding='same')(c2)    # 64 → 32

    c3 = conv_block(p2, t, 256)
    p3 = layers.Conv2D(256, 3, strides=2, padding='same')(c3)    # 32 → 16

    c4 = conv_block(p3, t, 512)
    p4 = layers.Conv2D(512, 3, strides=2, padding='same')(c4)    # 16 → 8

    # bottleneck — attention at 8x8 for global layout coherence
    bn = conv_block(p4, t, 512)
    bn = attention_block(bn)
    bn = conv_block(bn, t, 512)

    # decoder
    u4 = layers.UpSampling2D()(bn)
    u4 = layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, t, 256)                                   # 8 → 16
    c5 = attention_block(c5)                  # attention at 16x16 (fix #2)

    u3 = layers.UpSampling2D()(c5)
    u3 = layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, t, 256)                                   # 16 → 32

    u2 = layers.UpSampling2D()(c6)
    u2 = layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, t, 128)                                   # 32 → 64

    u1 = layers.UpSampling2D()(c7)
    u1 = layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, t, 64)                                    # 64 → 128

    # float32 output required for numerical stability with mixed precision (fix #4)
    outputs = layers.Conv2D(3, 1, padding='same', dtype='float32')(c8)

    return tf.keras.Model([x_input, t_input], outputs)