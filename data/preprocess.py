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
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, len(image_paths)


def preprocess_image(path, image_size):
    """
    Loads a single RICO screenshot and prepares it for training.
    - Center crops from 1440x2560 to 1440x1440 (preserves aspect ratio)
    - Resizes to image_size x image_size
    - Normalizes pixel values from [0, 255] to [-1, 1]
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3, try_recover_truncated=True)
    
    # rico images are 1440x2560 portrait — center crop to square
    # avoids squishing the aspect ratio when resizing
    image = tf.image.resize_with_crop_or_pad(image, 1440, 1440)
    image = tf.image.resize(image, [image_size, image_size])

    # cast and normalize to [-1, 1] for stable diffusion training
    image = tf.cast(image, tf.float32) / 255.0
    image = image * 2.0 - 1.0

    return image