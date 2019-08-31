import tensorflow as tf
import sys


class TFRecordHelper:
  def __init__(self, datasetName, pathToSave="", fileNamePrefix="", oneHotEncode = True):
    self.datasetName = datasetName
    self.pathToSave = pathToSave
    self.fileNamePrefix = fileNamePrefix
    self.trainDataShape = []
    self.oneHotEncode = oneHotEncode
    
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
    if self.datasetName.lower() == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        self.trainDataShape = [x_train.shape[1],x_train.shape[2],x_train.shape[3]]
        
        if self.pathToSave == "" :
          trainFileName = f'{self.fileNamePrefix}-train.tfrecords'
          testFileName = f'{self.fileNamePrefix}-test.tfrecords'
        else:
          trainFileName = f'{self.pathToSave}/{self.fileNamePrefix}-train.tfrecords'
          testFileName = f'{self.pathToSave}/{self.fileNamePrefix}-test.tfrecords'
        
        self._save_datarecord(trainFileName, x_train, y_train)
        self._save_datarecord(testFileName, x_test, y_test)
    return {'trainFileName':trainFileName,
            'testFileName': testFileName,
            'trainingDataShape': self.trainDataShape
    }
  
  
  def _parser(self, record):
    keys_to_features = {
        'image_raw': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed['image_raw'], tf.uint8)
    image = tf.cast(image, tf.int32)
    image = tf.reshape(image, [32,32,3])
    label = tf.cast(parsed["label"], tf.int32)
    print(f'One hot encode: {self.oneHotEncode}')
    if self.oneHotEncode:
      label = tf.one_hot(label, 10)
    return image, label
  
  def get_dataset_from_TFRecord(self, filename):
    dataset = tf.data.TFRecordDataset(filenames=filename)
    dataset = dataset.map(self._parser)
    return dataset