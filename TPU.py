import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
strategy = tf.distribute.experimental.TPUStrategy(resolver)
strategy.experimental_enable_dynamic_batch_size = False

def create_model():
  return tf.keras.Sequential(
      [tf.keras.layers.Conv1D(8,   4, activation='relu', input_shape=(8192,1)),
       tf.keras.layers.Conv1D(16,  4, activation='relu'),
       tf.keras.layers.Conv1D(64,  4, activation='relu'),
       tf.keras.layers.Conv1D(128, 4, activation='relu'),
       tf.keras.layers.Conv1D(128, 4, activation='relu'),
       tf.keras.layers.Conv1D(256, 4, activation='relu'),
       tf.keras.layers.Conv1D(256, 4, activation='relu'),
       tf.keras.layers.Conv1D(128, 4, activation='relu'),
       tf.keras.layers.Conv1D(64,  4, activation='relu'),
       tf.keras.layers.Conv1D(16,  4, activation='relu'),
       tf.keras.layers.Conv1D(8,   4, activation='relu'),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(256, activation='relu'),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(44)])


data=np.random.random((60000,8192))
labels=np.random.random((60000,44))
for j in range(labels.shape[0]):
  for i in range(labels.shape[1]):
    if labels[j][i]>0.5:
      labels[j][i]=1
    else :
      labels[j][i]= 0 

train_examples, train_labels = data[:50000],labels[:50000]
test_examples, test_labels = data[50001:],labels[50001:]

train_examples=train_examples.reshape((train_examples.shape[0],train_examples.shape[1],1))
test_examples= test_examples.reshape((test_examples.shape[0],test_examples.shape[1],1))
print(train_examples.shape, train_labels.shape, test_examples.shape,train_labels.shape)

train_examples = tf.cast(train_examples,tf.float32)
train_labels = tf.cast(train_labels,tf.float32)
test_examples = tf.cast(test_examples,tf.float32)
test_labels = tf.cast(test_labels,tf.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

#train_dataset =(train_examples, train_labels)
#test_dataset = (test_examples, test_labels)

train_dataset = train_dataset.shuffle(10000).batch(1000)
train_dataset = train_dataset.repeat()
test_dataset = test_dataset.batch(200)

with strategy.scope():
  model = create_model()
  model.compile(optimizer='SGD',
                loss=tf.keras.losses.binary_crossentropy,
                metrics=['binary_accuracy'])

batch_size = 200
steps_per_epoch = 60000 // batch_size

model.fit(train_dataset,
          epochs=5,
          steps_per_epoch=steps_per_epoch,
          validation_data=test_dataset)
