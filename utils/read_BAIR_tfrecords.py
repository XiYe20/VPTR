import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
import os
from tqdm import tqdm

#Requirements: tensorflow 2.6.0
def read_BAIR_tf2_record(records_dir, save_dir):
    """
    Args:
        record_file: string for the BAIR tf record file path
    Returns:
        imgs: The images saved in the input record_file
    """
    ORIGINAL_HEIGHT = 64
    ORIGINAL_WIDTH = 64
    COLOR_CHAN = 3

    records_path = Path(records_dir)
    tf_record_files = sorted(list(records_path.glob('*.tfrecords')))
    dataset = tf.data.TFRecordDataset(tf_record_files)

    pgbar = tqdm(total = 256*len(tf_record_files), desc = 'Processing...')
    for example_id, example in enumerate(dataset):
        example_dir = Path(save_dir).joinpath(f'example_{example_id}')
        if example_dir.exists():
            shutil.rmtree(example_dir.absolute())
        example_dir.mkdir(parents=True, exist_ok=True)

        for i in range(0, 30):
            image_main_name = str(i) + '/image_main/encoded'
            image_aux1_name = str(i) + '/image_aux1/encoded'

            features = {image_aux1_name: tf.io.FixedLenFeature([1], tf.string),
                        image_main_name: tf.io.FixedLenFeature([1], tf.string)}
            
            features = tf.io.parse_single_example(example, features=features)

            image_aux1 = tf.io.decode_raw(features[image_aux1_name], tf.uint8)
            image_aux1 = tf.reshape(image_aux1, shape=[1, ORIGINAL_HEIGHT * ORIGINAL_WIDTH * COLOR_CHAN])
            image_aux1 = tf.reshape(image_aux1, shape=[ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])

            frame_name = example_dir.joinpath(f'{i:04n}.png')
        
            frame = Image.fromarray(image_aux1.numpy(), 'RGB')
            frame.save(frame_name.absolute().as_posix())
            #frame = tf.image.encode_png(image_aux1)
            #with open(frame_name.absolute().as_posix(), 'wb') as f:
            #    f.write(frame)
        pgbar.update(1)

def resize_im(features, image_name, conf, height = None):
    COLOR_CHAN = 3
    if '128x128' in conf:
        ORIGINAL_WIDTH = 128
        ORIGINAL_HEIGHT = 128
        IMG_WIDTH = 128
        IMG_HEIGHT = 128
    elif height != None:
        ORIGINAL_WIDTH = height
        ORIGINAL_HEIGHT = height
        IMG_WIDTH = height
        IMG_HEIGHT = height
    else:
        ORIGINAL_WIDTH = 64
        ORIGINAL_HEIGHT = 64
        IMG_WIDTH = 64
        IMG_HEIGHT = 64

    image = tf.decode_raw(features[image_name], tf.uint8)
    image = tf.reshape(image, shape=[1, ORIGINAL_HEIGHT * ORIGINAL_WIDTH * COLOR_CHAN])
    image = tf.reshape(image, shape=[ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
    if IMG_HEIGHT != IMG_WIDTH:
        raise ValueError('Unequal height and width unsupported')
    crop_size = min(ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
    image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
    image = tf.reshape(image, [1, crop_size, crop_size, COLOR_CHAN])
    image = tf.image.resize_bicubic(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0

    return image

if __name__ == '__main__':
    """
    read_BAIR_tf2_record('BAIR/softmotion30_44k/test/traj_0_to_255.tfrecords',
                   'BAIR/softmotion30_44k/test')
    """
    
    read_BAIR_tf2_record('BAIR/softmotion30_44k/train', 'BAIR/softmotion30_44k/train')
