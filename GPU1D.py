#GPU

#pip install gwpy lalsuite Pycbc

import numpy as np
#from pycbc.waveform import get_td_waveform
#from gwpy.timeseries import TimeSeries
from tqdm import tqdm
#---Data reading and writing---------------
import csv
import h5py
import pandas as pd
from scipy import signal
import scipy.io.wavfile as s
import matplotlib.pyplot as plt

#signal_gw=np.zeros((135600,8192))
#next_val=0

from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
root_dir = "/content/gdrive/My Drive/"
base_dir = root_dir + 'fastai-v3/'

ls gdrive/My\ Drive/GW\ data/

hf= h5py.File('gdrive/My Drive/GW data/mixed.h5', 'r')
group_key = list(hf.keys())[0]
signal_gw= hf[group_key]
signal_gw=np.array(signal_gw)
print(signal_gw.shape,type(signal_gw))
hf.close()

labels=pd.read_csv('gdrive/My Drive/GW data/labels_mixed.csv')
labels.shape

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split

data=signal_gw#[:next_val]
print(data.shape)
data=data.reshape((data.shape[0],data.shape[1],1))
print(data.shape)

labels=pd.read_csv('gdrive/My Drive/GW data/labels_mixed.csv')
labels=labels.to_numpy(dtype=int,copy=True)

X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=1, test_size=0.2)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

model = Sequential()
model.add(Conv1D(filters=4,   kernel_size=4, activation="relu", input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(Conv1D(filters=8,   kernel_size=4, activation="relu", input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(Conv1D(filters=16,  kernel_size=4, activation="relu",padding='valid',use_bias='True'))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32,  kernel_size=4, activation='relu',padding='valid',use_bias='True'))
model.add(Dropout(0.3))
model.add(Conv1D(filters=64,  kernel_size=4, activation="relu",padding='valid',use_bias='True'))
model.add(Dropout(0.3))
model.add(Conv1D(filters=128, kernel_size=4, activation="relu",padding='valid',use_bias='True'))
model.add(Dropout(0.3))
model.add(Conv1D(filters=256, kernel_size=4, activation="relu",padding='valid',use_bias='True'))
model.add(Dropout(0.3))
model.add(Conv1D(filters=512, kernel_size=4, activation="relu",padding='valid',use_bias='True'))
model.add(Dropout(0.3))
model.add(Conv1D(filters=512, kernel_size=4, activation="relu",padding='valid',use_bias='True'))
model.add(Dropout(0.3))
model.add(Conv1D(filters=256, kernel_size=4, activation="relu",padding='valid',use_bias='True'))
model.add(Dropout(0.3))
model.add(Conv1D(filters=128, kernel_size=4, activation="relu",padding='valid',use_bias='True'))
model.add(Dropout(0.2))
model.add(Conv1D(filters=64,  kernel_size=4, activation="relu",padding='valid',use_bias='True'))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32,  kernel_size=4, activation="relu",padding='valid',use_bias='True'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(44, activation='sigmoid'))

sgd = SGD(lr=1e-3,decay=1e-6, momentum=0.95,nesterov=True, clipnorm=1.0)
#sgd = Adam(lr=1e-6,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['binary_accuracy','accuracy'])
a=model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=50)

model.save('gdrive/My Drive/GW data/my_model',save_format='h5')

plt.plot(a.history['accuracy'])
plt.plot(a.history['val_accuracy'])
plt.plot(a.history['loss'])
plt.plot(a.history['val_loss'])
plt.xlabel('epoch')
plt.legend(['accuracy', 'val_accuracy','loss','val_loss'], loc='upper left')
plt.figure(figsize=(20,10))
plt.show()

model.predict(X_train[5])
print(labels[5])

#from keras.models import model_from_json
#model_json = model.to_json()
#with open("gdrive/My Drive/GW data/model.json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#model.save_weights("gdrive/My Drive/GW data/model.h5")
#print("Saved model to disk")

!a
