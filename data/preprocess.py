import tensorflow as tf

def get_dataset(data_path, image_size, batch_size):
    """
    Loads RICO screenshots from Drive, preprocesses them,
    and returns a batched tf.data pipeline ready for training.
    """

    # glob handles large Drive directories reliably
    # os.listdir fails with I/O errors on folders with tens of thousands of files
    image_paths = tf.io.gfile.glob(data_path + '/*.jpg')
    print(f'Total UI screenshots found: {len(image_paths)}')

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.shuffle(buffer_size=5000, seed=42)
    dataset = dataset.map(
        lambda path: preprocess_image(path, image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # preprocess_image returns (image, valid) — drop any files that were empty
    # the filter only checks a boolean, no extra Drive reads
    dataset = dataset.filter(lambda img, valid: valid)
    dataset = dataset.map(lambda img, valid: img)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, len(image_paths)


def preprocess_image(path, image_size):
    """
    Loads a single RICO screenshot and prepares it for training.
    Returns (image, valid) so the pipeline can filter empty files
    without a separate read pass.
    - Center crops from 1440x2560 to 1440x1440 (preserves aspect ratio)
    - Resizes to image_size x image_size
    - Normalizes pixel values from [0, 255] to [-1, 1]
    """
    raw = tf.io.read_file(path)
    valid = tf.math.greater(tf.strings.length(raw), 100)

    def decode_and_process():
        img = tf.image.decode_jpeg(raw, channels=3, try_recover_truncated=True)
        # rico images are 1440x2560 portrait — center crop to square
        # avoids squishing the aspect ratio when resizing
        img = tf.image.resize_with_crop_or_pad(img, 1440, 1440)
        img = tf.image.resize(img, [image_size, image_size])
        # cast and normalize to [-1, 1] for stable diffusion training
        img = tf.cast(img, tf.float32) / 255.0
        return img * 2.0 - 1.0

    image = tf.cond(
        valid,
        decode_and_process,
        lambda: tf.zeros([image_size, image_size, 3], dtype=tf.float32)
    )

    return image, valid
