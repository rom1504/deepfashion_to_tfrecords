import tensorflow as tf

def read_raw_dataset(root_path):
    annos = tf.data.Dataset.list_files(root_path+"/annos/*.json").map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    images = tf.data.Dataset.list_files(root_path+"/image/*.jpg").map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    both = tf.data.Dataset.zip((annos, images))
    return both

def generate_captions(classname):
    return [
        f"Here you see an image that contains {classname}.".encode('utf-8'),
        f"An image that contains {classname}.".encode('utf-8'),
        f"Here you see a {classname}".encode('utf-8'),
        f"This is a {classname}".encode('utf-8'),
        f"Here you see a {classname}".encode('utf-8'),
        f"{classname}".encode('utf-8'),
    ]

def process_path(file_path):
    parts = tf.strings.split(file_path, "/")
    image_name = tf.strings.split(parts[-1], ".")[0]
    raw = tf.io.read_file(file_path)
    return raw, image_name

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_list_feature(values):
    """Returns a bytes_list from a string / byte."""
    if isinstance(values, type(tf.constant(0))):
        values = values.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

item_keys = ['item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7', 'item8', 'item9', 'item10']
import itertools
import fire

def serialize_example(annos, image):
    import json
    image, image_name = image
    jjson, _ = annos
    decoded = json.loads(jjson.numpy().decode('utf-8'))
    img_tensor = tf.image.decode_jpeg(image)
    width = img_tensor.shape[0]
    height = img_tensor.shape[1]

    categories = []
    for i in item_keys:
        if i not in decoded:
            break
        categories.append(decoded[i]['category_name'])
    captions = list(itertools.chain.from_iterable([generate_captions(c) for c in categories]))
    
    feature = {"image_name": _bytes_feature(image_name),
               "image": _bytes_feature(image),
               "height": _int64_feature(height),
               "width": _int64_feature(width),
               "captions": _bytes_list_feature(captions),
              }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(annos, image):
    tf_string = tf.py_function(serialize_example, (annos, image), tf.string)
    return tf.reshape(tf_string, ())

import os
import time

def main_converter(dataset_path, num_shards=100, output_folder="output_folder", limit_per_shard=None) :
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    both = read_raw_dataset(dataset_path)
    
    for i in range(num_shards):
        start = time.time()
        print("Starting shard "+str(i))
        shard = both.shard(num_shards=num_shards, index=i)
        serialized_features_dataset = shard.map(tf_serialize_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        limited_dataset = serialized_features_dataset.take(limit_per_shard) if limit_per_shard is not None else serialized_features_dataset
        
        writer = tf.data.experimental.TFRecordWriter(output_folder+f"/part_{i:03d}.tfrecord")
        writer.write(limited_dataset)
        duration = time.time() - start
        print(f"Shard {str(i)} done in {str(duration)}s")
        
if __name__ == '__main__':
    fire.Fire(main_converter)
