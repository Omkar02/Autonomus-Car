from __future__ import division, print_function, absolute_import
from keras.models import Model, load_model
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.models import Sequential
import h5py
import os
from keras import callbacks
from scipy import misc
from keras.utils import plot_model
from keras import losses
from skimage import io,transform,color
from keras import metrics
import cv2
import os, sys
from PIL import Image

model = load_model(model_path)
#***************LOAD_DATA*********************************************
with open('indoor_lanes.pkl', 'rb') as f:
    X, Y = pickle.load(f)

print('X.shape: ', X.shape)
print('Y.shape: ', Y.shape)

#****************shuffle  both X and Y the same way********************************************
def unison_shuffled_copies(X, Y):
    assert len(X) == len(Y)
    p = np.random.permutation(len(X))
    return X[p], Y[p]

shuffled_X, shuffled_Y = unison_shuffled_copies(X,Y)

len(shuffled_X)

test_cutoff = int(len(X) * .8)                          # 80% of data used for training
val_cutoff = test_cutoff + int(len(X) * .1)             # 10% of data used for validation and test data

train_X, train_Y = shuffled_X[:test_cutoff], shuffled_Y[:test_cutoff]
val_X, val_Y = shuffled_X[test_cutoff:val_cutoff], shuffled_Y[test_cutoff:val_cutoff]
test_X, test_Y = shuffled_X[val_cutoff:], shuffled_Y[val_cutoff:]

len(train_X) + len(val_X) + len(test_X)



X_flipped = np.array([np.fliplr(i) for i in train_X])
Y_flipped = np.array([-i for i in train_Y])
train_X = np.concatenate([train_X, X_flipped])
print(np.shape(train_X))
train_Y = np.concatenate([train_Y, Y_flipped])
len(train_X)

#****************INPUT_LAYER)***************************************
img_in = Input(shape=(120, 160, 3), name='img_in')
angle_in = Input(shape=(1,), name='angle_in')
#****************LAYER_1********************************************
x = Convolution2D(8, 3, 3)(img_in)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
#****************LAYER_2********************************************
x = Convolution2D(16, 3, 3)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
#****************LAYER_3********************************************
x = Convolution2D(32, 3, 3)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

merged = Flatten()(x)

x = Dense(512)(merged)
x = Activation('linear')(x)
x = Dropout(.2)(x)

angle_out = Dense(1, name='angle_out')(x)
#****************MODEL_DETAIL********************************************
model = Model(input=[img_in], output=[angle_out])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
model.summary()
#****************OUTPUT_PATH********************************************
model_path = 'best_autopilot.hdf5'

#Save the model after each epoch if the validation loss improved.
save_best = callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')

#stop training if the validation loss doesn't improve for 5 consecutive epochs.
early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2,
                                     verbose=0, mode='auto')

callbacks_list = [save_best, early_stop]

model.fit(train_X, train_Y, batch_size=64, nb_epoch=30, validation_data=(val_X, val_Y), callbacks=callbacks_list)
#***************** TEST **********************************************************************************
score = model.evaluate(test_X, test_Y)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(test_Y[1:5])

#*****************PREDICTION****************************************************************************
# model = load_model(model_path)
# data_pa='testing'
# a=0
# x = []
# for dataset in os.listdir(data_pa):
#     a+=1
#     img = Image.open(os.path.join(data_pa, dataset))
#     img = img.resize((120, 160), Image.ANTIALIAS)
#     img = np.reshape(img, (-1,120, 160, 3))
#     test = model.predict(img)
#     x.append(float(test))
#     print(a,'.Angle = ',test)
#
#
# Y =np.arange(a)
# plt.figure(1)
# plt.plot(Y,x)
# plt.savefig('PID_te1.png')
#*****************PREDICTION****************************************************************************




