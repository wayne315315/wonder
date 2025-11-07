import tensorflow as tf


def _bytes_feature(value_tensor):
    """Returns a bytes_list from a serialized tensor."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[
        tf.io.serialize_tensor(value_tensor).numpy()
    ]))

def create_example(v, h, y, r, l):
    feature = {
        'vs': _bytes_feature(v),
        'hs': _bytes_feature(h),
        'ys': _bytes_feature(y),
        'rs': _bytes_feature(r),
        'ls': _bytes_feature(l),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def parse_example(example_proto):
    feature_description = {
        'vs': tf.io.FixedLenFeature([], tf.string),
        'hs': tf.io.FixedLenFeature([], tf.string),
        'ys': tf.io.FixedLenFeature([], tf.string),
        'rs': tf.io.FixedLenFeature([], tf.string),
        'ls': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    v = tf.io.parse_tensor(parsed_features['vs'], out_type=tf.int32)
    h = tf.io.parse_tensor(parsed_features['hs'], out_type=tf.int32)
    y = tf.io.parse_tensor(parsed_features['ys'], out_type=tf.int32)
    r = tf.io.parse_tensor(parsed_features['rs'], out_type=tf.float32)
    l = tf.io.parse_tensor(parsed_features['ls'], out_type=tf.float32)
    #v.set_shape([None, None, 7])
    #h.set_shape([None, None])
    #y.set_shape([None])
    #r.set_shape([None])
    #l.set_shape([None, 231])
    return (v, h, y, r, l)


if __name__ == "__main__":
    from pathlib import Path

    v = tf.ones((369, 120, 7), dtype=tf.int32)
    h = tf.ones((369, 7), dtype=tf.int32)
    y = tf.ones((369,), dtype=tf.int32)
    r = tf.ones((369,), dtype=tf.float32)
    l = tf.ones((369, 231), dtype=tf.float32)
    example = create_example(v, h, y, r, l)

    output_path_1 = Path("generated_data.tfrecord")
    with tf.io.TFRecordWriter(str(output_path_1)) as writer:
        writer.write(example.SerializeToString())

    v = tf.ones((36, 63, 7), dtype=tf.int32)
    h = tf.ones((36, 21), dtype=tf.int32)
    y = tf.ones((36,), dtype=tf.int32)
    r = tf.ones((36,), dtype=tf.float32)
    l = tf.ones((36, 231), dtype=tf.float32)
    example = create_example(v, h, y, r, l)

    output_path_2 = Path("generated_data_2.tfrecord")
    with tf.io.TFRecordWriter(str(output_path_2)) as writer:
        writer.write(example.SerializeToString())

    dataset = tf.data.TFRecordDataset([output_path_1, output_path_2])
    dataset = dataset.map(parse_example)
    for v_parsed, h_parsed, y_parsed, r_parsed, l_parsed in dataset:
        print(v_parsed.shape, h_parsed.shape, y_parsed.shape, r_parsed.shape, l_parsed.shape)
