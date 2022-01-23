from __future__ import division, print_function, absolute_import
from keras.models import Model, load_model
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.models import model_from_json
from keras.models import Sequential
import h5py
from keras import callbacks
from scipy import misc
from keras.utils import plot_model
from keras import losses
from skimage import io,transform,color
from keras import metrics
import cv2
import os, sys
from PIL import Image
from sklearn.utils import shuffle
import time
#***************LOAD_DATA*********************************************
data= ('testing')
list = os.listdir(data) # dir is your directory path
number_files = len(list)
label2 = np.arange(number_files)

#***************LOAD_DATA*********************************************
with open('indoor_lanes.pkl', 'rb') as f:
    X,Y = pickle.load(f)


print('X.shape: ', X.shape)
print('Y.shape: ', Y.shape)

with open('testing_img.pkl', 'rb') as f:
    C = pickle.load(f)
    print(len(C))
    V = label2

print('C.shape: ', C.shape)

train_X, train_Y = (X, Y)
train_X, train_Y = shuffle(train_X, train_Y, random_state=2)
val_X, val_Y = (C, V)
val_X, val_Y = shuffle(val_X, val_Y, random_state=3)
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
#****************LAYER_4********************************************
x = Convolution2D(64, 3, 3)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
#****************LAYER_5********************************************
x = Convolution2D(128, 3, 3)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)


merged = Flatten()(x)

x = Dense(256)(merged)
x = Activation('linear')(x)
x = Dropout(.2)(x)

angle_out = Dense(1, name='angle_out')(x)
#****************MODEL_DETAIL********************************************
model = Model(input=[img_in], output=[angle_out])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
#****************OUTPUT_PATH********************************************
model_path = 'model/best_autopilot_new.hdf5'

#Save the model after each epoch if the validation loss improved.
save_best = callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')

#stop training if the validation loss doesn't improve for 5 consecutive epochs.
early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20,
                                     verbose=0, mode='auto')

callbacks_list = [save_best, early_stop]

model.fit(train_X, train_Y, batch_size=64, nb_epoch=23, validation_data=(val_X, val_Y),
          callbacks=callbacks_list)
# score = model.evaluate(test_X, test_Y)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
#****************PREDICTION********************************************
model = load_model(model_path)
data_pa='training'
x = []
for dataset in os.listdir(data_pa):
    img = Image.open(os.path.join(data_pa, dataset))
    img = img.resize((120, 160), Image.ANTIALIAS)
    img = np.reshape(img, (-1,120, 160, 3))
    test = model.predict(img)
    x.append(float(test))
    print('Angle = ',test)
    #time.sleep(1)
Y =np.arange(290)
plt.figure(1)
plt.plot(Y,x)
plt.savefig('NN112_te.png')