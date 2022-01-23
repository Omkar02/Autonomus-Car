from __future__ import division, print_function, absolute_import
from keras.models import Model, load_model
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
import numpy as np
import pickle
from keras import callbacks
import os
import matplotlib.pyplot as plt
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
import RPi.GPIO as GPIO
from time import sleep

lis = []

# ***************LOAD_DATA*********************************************
# with open('indoor_lanes.pkl', 'rb') as f:
#    X, Y = pickle.load(f)

# print('X.shape: ', X.shape)
# print('Y.shape: ', Y.shape)

# ****************shuffle  both X and Y the same way********************************************
# def unison_shuffled_copies(X, Y):
#   assert len(X) == len(Y)
#    p = np.random.permutation(len(X))
#    return X[p], Y[p]

# shuffled_X, shuffled_Y = unison_shuffled_copies(X,Y)

# len(shuffled_X)

# test_cutoff = int(len(X) * .8)                          # 80% of data used for training
# val_cutoff = test_cutoff + int(len(X) * .1)             # 10% of data used for validation and test data

# train_X, train_Y = shuffled_X[:test_cutoff], shuffled_Y[:test_cutoff]
# val_X, val_Y = shuffled_X[test_cutoff:val_cutoff], shuffled_Y[test_cutoff:val_cutoff]
# test_X, test_Y = shuffled_X[val_cutoff:], shuffled_Y[val_cutoff:]

# len(train_X) + len(val_X) + len(test_X)



# X_flipped = np.array([np.fliplr(i) for i in train_X])
# Y_flipped = np.array([-i for i in train_Y])
# train_X = np.concatenate([train_X, X_flipped])
# print(np.shape(train_X))
# train_Y = np.concatenate([train_Y, Y_flipped])
# len(train_X)

# ****************INPUT_LAYER)***************************************
img_in = Input(shape=(120, 160, 3), name='img_in')
angle_in = Input(shape=(1,), name='angle_in')
# ****************LAYER_1********************************************
x = Convolution2D(8, 3, 3)(img_in)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
# ****************LAYER_2********************************************
x = Convolution2D(16, 3, 3)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
# ****************LAYER_3********************************************
x = Convolution2D(32, 3, 3)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
# ****************LAYER_4********************************************
x = Convolution2D(64, 3, 3)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
# ****************LAYER_5********************************************
x = Convolution2D(128, 3, 3)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

merged = Flatten()(x)

x = Dense(256)(merged)
x = Activation('linear')(x)
x = Dropout(.2)(x)

angle_out = Dense(1, name='angle_out')(x)
# ****************MODEL_DETAIL********************************************
model = Model(input=[img_in], output=[angle_out])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
model.summary()
# ****************OUTPUT_PATH********************************************
model_path = 'best_autopilot_new.hdf5'

# Save the model after each epoch if the validation loss improved.
save_best = callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=1,
                                      save_best_only=True, mode='min')

# stop training if the validation loss doesn't improve for 5 consecutive epochs.
early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                                     verbose=0, mode='auto')

callbacks_list = [save_best, early_stop]

#model.fit(train_X, train_Y, batch_size=64, nb_epoch=30, validation_data=(val_X, val_Y), callbacks=callbacks_list)
# ***********************************************************************
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    cv2.imshow('St',image)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    imag = cv2.resize(image, (120, 160))
    # print(imag.shape)
    imag = np.reshape(imag, (-1, 120, 160, 3))
    # print(imag.shape)

    value = model.predict(imag)
    #print('Value = ', value)
    # ****************************************************************************
    y = value

    yc = 400
    h = 1

    Ti = 1
    Td = 0.48
    Kp = 0.4
    u0 = 0
    e0 = 0

    """Calculate System Input using a PID Controller

    Arguments:
    y  .. Measured Output of the System
    yc .. Desired Output of the System
    h  .. Sampling Time
    Kp .. Controller Gain Constant
    Ti .. Controller Integration Constant
    Td .. Controller Derivation Constant
    u0 .. Initial state of the integrator
    e0 .. Initial error

    Make sure this function gets called every h seconds!
    """

    # Step variable
    k = 0

    # Initialization
    ui_prev = u0
    e_prev = e0
    x = []

    y = y + 1
    # Error between the desired and actual output
    e = yc - y
    # Integration Input
    ui = ui_prev + 1 / Ti * h * e
    # Derivation Input
    ud = 1 / Td * (e - e_prev) / h
    # Adjust previous values
    e_prev = e
    ui_prev = ui
    # Calculate input for the system
    u = Kp * (e + ui + ud)

    GPIO.setmode(GPIO.BOARD)

    GPIO.setup(37, GPIO.OUT)
    GPIO.setup(35, GPIO.OUT)
    GPIO.setup(33, GPIO.OUT)
    GPIO.setup(31, GPIO.OUT)

    GPIO.output(37, True)
    GPIO.output(35, True)

    if u >= 0:
        print('R = ',value)
        GPIO.output(33, True)
        GPIO.output(31, False)

    elif u <= 0:
   	    print('L = ',value)
        GPIO.output(33, False)
        GPIO.output(31, True)
    elif key == ord('q'):

	    GPIO.output(33, False)
        GPIO.output(31, False)
        break

    else:
        sleep(0.5)
        GPIO.output(33, False)
        GPIO.output(31, False)