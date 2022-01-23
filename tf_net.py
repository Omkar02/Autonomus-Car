from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected,flatten
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
#**************************************************************************************** Data loading and preprocessing
with open('indoor_lanes.pkl', 'rb') as f:
    X,Y = pickle.load(f)
number_files = len(X)

Y_data = np.asarray(Y)
Y = Y_data.reshape(number_files,1)
Y = Y.astype('float32')
print('Array Y = ',Y)

print('X.shape: ', X.shape)
print('Y.shape: ', Y.shape)
#**************************************************************************************** Convolutional network building
# net = tflearn.input_data(shape=[None, 120, 160, 3], name='input')
# net = tflearn.conv_2d(net, 8, strides=2, activation='relu')
# net = tflearn.conv_2d(net, 16, strides=2, activation='relu')
# net = tflearn.conv_2d(net, 32, strides=2, activation='relu')
# net = tflearn.conv_2d(net, 64, activation='relu')
# net = tflearn.conv_2d(net, 128, activation='relu')
# net = tflearn.fully_connected(net, 256, activation='relu')
# net = tflearn.dropout(net, 0.5)
# # net = tflearn.fully_connected(net, 64, activation='relu')
# # net = tflearn.dropout(net, 0.5)
# # net = tflearn.fully_connected(net, 32, activation='relu')
# # net = tflearn.dropout(net, 0.5)
# # net = tflearn.fully_connected(net, 8, activation='relu')
# # net = tflearn.dropout(net, 0.5)
network = input_data(shape=[None, 120, 160, 3])
network = conv_2d(network, 192, 5, activation='relu')
network = conv_2d(network, 160, 1, activation='relu')
network = conv_2d(network, 96, 1, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = dropout(network, 0.5)
network = conv_2d(network, 192, 5, activation='relu')
network = conv_2d(network, 192, 1, activation='relu')
network = conv_2d(network, 192, 1, activation='relu')
network = avg_pool_2d(network, 3, strides=2)
network = dropout(network, 0.5)
network = conv_2d(network, 192, 3, activation='relu')
network = conv_2d(network, 192, 1, activation='relu')
network = conv_2d(network, 36, 1, activation='relu')
network = avg_pool_2d(network, 8)
network = flatten(network)
network = tflearn.fully_connected(network, 1, activation='tanh')
network = regression(network, optimizer='adam',
                     loss='softmax_categorical_crossentropy',
                     learning_rate=0.001)
#************************************************************************************************ Train using classifier
model = tflearn.DNN(network)
model.fit(X, Y, n_epoch=20, shuffle=True,
          show_metric=True, batch_size=128, run_id='Detect it yarr!')
model.save('model/model_carrTf_new.tflearn')
# #******************* predict *******************************************************************************************
model.load('model/model_carrTf_new.tflearn')
data_pa='testing'
x = []

d = 0
for dataset in os.listdir(data_pa):
    img = Image.open(os.path.join(data_pa, dataset))
    img = img.resize((120, 160), Image.ANTIALIAS)
    img = np.reshape(img, (-1,120, 160, 3))
    test = model.predict(img)
    x.append(float(test))
    print('%d_Angle = '%d,test)
    d+=1
    #time.sleep(1)
Y =np.arange(290)
plt.figure(1)
plt.plot(Y,x)
plt.savefig('NN_te.png')



