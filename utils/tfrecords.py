"""
    - Lets write functions to do this automatically
"""
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

def convert_np_to_tf_example(image, label):
    feature_dict = {
        'image': _bytes_feature(image.tostring()),
        'label': _int64_feature(label),
    }

    features = tf.train.Features(feature=feature_dict)
    tf_example = tf.train.Example(features=features)
    protocol_message = tf_example.SerializeToString()

    return protocol_message

def convert_proto_message_to_np(protocol_message):
    feature_dict = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    parsed_out = tf.io.parse_single_example(serialized=protocol_message,
                                            features=feature_dict)
    
    image = tf.io.decode_raw(parsed_out['image'], out_type=tf.uint8)
    label = parsed_out['label']

    return (image,label)
    
def reshape_img(X, img_shape):
    img = X[0].numpy()
    label = X[1]
    img = img.reshape(img_shape)
    img_tensor = tf.convert_to_tensor(img)

    return (img_tensor, label)
    

def convert_np_to_tfrecords(images, labels, 
                            batch_size=128,filename=None):
    """        for img in (images,labels):

        Converts a numpy array into TFReocrds
    """
    assert images.shape[0] == labels.shape[0], " Number of Images is not equal to number of labels"

    channels = images.shape[-1]
    width = images.shape[-2]
    height = images.shape[-3]

    img_shape = (height, width, channels)
    if filename == None:
        filename = 'dataset.tfrecords'
    with tf.io.TFRecordWriter(filename) as writer:
        for img,label in zip(images,labels):
            protocol_message = convert_np_to_tf_example(img,label)

            writer.write(protocol_message)
        
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(convert_proto_message_to_np)
    dataset = dataset.map(lambda x, y: (tf.reshape(x,img_shape), y))
    dataset = dataset.batch(batch_size)
    return dataset

dataset = convert_np_to_tfrecords(X_train[0],X_train[1])
