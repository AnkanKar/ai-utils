import tensorflow as tf
import sys
from tensorflow.keras.datasets import cifar10


class TFRecordHelper:
  def __init__(self, dataset_name, path_to_save="", file_name_prefix="", one_hot_encode = True):
    """TFRecordHelper is a class made to ease the use of TFRecords with Keras, It writes data into TFRecords,
       and Reads the data from them as well, in a format directly understandle by your model
    
    Arguments:
        dataset_name {[type]} -- Class has a few datasets inbuilt, Pass name of dataset to use eg. cifar10
    
    Keyword Arguments:
        path_to_save {str} -- Path to directory where to save TFRecords  (default: {""})
        file_name_prefix {str} -- Prefix used for Train and test TFRecord files (default: {""})
        one_hot_encode {bool} -- Used to check wheather to one hot encode Labels (default: {True})
    """
    
    self.dataset_name = dataset_name
    self.path_to_save = path_to_save
    self.file_name_prefix = file_name_prefix
    self.train_data_shape = []
    self.one_hot_encode = one_hot_encode
    
  def _int64_feature(self, value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def _bytes_feature(self, value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  
  def _save_datarecord(self, out_filename, images, labels):
  
    with tf.io.TFRecordWriter(out_filename) as writer:
    
        for i in range(len(images)):
          feature = {
              'image_raw': self._bytes_feature(images[i].tostring()),
              'label': self._int64_feature(labels[i])
          }

          example = tf.train.Example(features=tf.train.Features(feature=feature))

          writer.write(example.SerializeToString())

    sys.stdout.flush()
  
  def create_tfrecord(self):
    """Creates test and train TFRecord files for the dataset provided during class initialization
    
    Returns:
        dict -- dictionary the contains file names of test, train tfrecod files and shape of training data
    """
    if self.dataset_name.lower() == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        self.train_data_shape = [x_train.shape[1],x_train.shape[2],x_train.shape[3]]
        
        if self.path_to_save == "" :
          train_file_name = f'{self.file_name_prefix}-train.tfrecords'
          test_file_name = f'{self.file_name_prefix}-test.tfrecords'
        else:
          train_file_name = f'{self.path_to_save}/{self.file_name_prefix}-train.tfrecords'
          test_file_name = f'{self.path_to_save}/{self.file_name_prefix}-test.tfrecords'
        
        self._save_datarecord(train_file_name, x_train, y_train)
        self._save_datarecord(test_file_name, x_test, y_test)
    return {'trainFileName':train_file_name,
            'testFileName': test_file_name,
            'trainingDataShape': self.train_data_shape
    }
  
  
  def _parser(self, record):
    """
    parses dats from a TFDataset
    """
    keys_to_features = {
        'image_raw': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed['image_raw'], tf.uint8)
    image = tf.cast(image, tf.int32)
    image = tf.reshape(image, [32,32,3])
    label = tf.cast(parsed["label"], tf.int32)
    if self.one_hot_encode:
      label = tf.one_hot(label, 10)
    return image, label
  
  def get_dataset_from_TFRecord(self, filename):
    """Get paresed Data set from TFRecods file
    
    Arguments:
        filename {str} -- file name of the TFRecord file you want to read    
    Returns:
        TFAdapter -- This can be directly passed to your model
    """
    dataset = tf.data.TFRecordDataset(filenames=filename)
    dataset = dataset.map(self._parser)
    return dataset
