import os
import tqdm
import glob
import random
import tensorflow as tf
import argparse
import sys

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example(img_str, source_id, filename):
    # Create a dictionary with features that may be relevant.
    feature = {'image/source_id': _int64_feature(source_id),
               'image/filename': _bytes_feature(filename),
               'image/encoded': _bytes_feature(img_str)}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(args):
    dataset_path = args.dataset_path

    if not os.path.isdir(dataset_path):
        print('Please define valid dataset path.')
    else:
        print('Loading {}'.format(dataset_path))


    # create map for idnum and idname
    id_name_list = os.listdir(dataset_path)

    samples = []
    print('Reading data list...')
    for idx in tqdm.tqdm(range(len(id_name_list))):
        img_paths = glob.glob(os.path.join(dataset_path, id_name_list[idx], '*.jpg'))
        for img_path in img_paths:
            filename = os.path.join(id_name_list[idx], os.path.basename(img_path))
            samples.append((img_path, idx, filename))
    random.shuffle(samples)

    print('Writing tfrecord file...')
    with tf.io.TFRecordWriter(args.output_path) as writer:
        for img_path, idx, filename in tqdm.tqdm(samples):
            tf_example = make_example(img_str=open(img_path, 'rb').read(),
                                      source_id=int(idx),
                                      filename=str.encode(filename))
            writer.write(tf_example.SerializeToString())


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
        help='path to dataset', default='../Dataset/ms1m_align_112/imgs')
    parser.add_argument('--output_path', type=str,
    help='path to ouput tfrecord', default='./data/ms1m_bin.tfrecord')
    parser.add_argument('--output_idmap_path', type=str,
    help='path to ouput tfrecord idmap', default='./data/ms1m_bin.tfrecord.idmap')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))

