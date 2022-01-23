import numpy as np
import os
import pickle
from PIL import Image

#**************************************************************************
#file_Name = "training_img.pkl"
file_Name = "testing_img.pkl"
img_data_list=[]
#**************************************************************************
data_path='testing'
#data_path='training'
data_dir_list = os.listdir(data_path)
#**************************************************************************
for dataset in os.listdir(data_path):

	img = Image.open(os.path.join(data_path, dataset))
	img = img.resize((120, 160), Image.ANTIALIAS)
	img = np.reshape(img, (120, 160, 3))
	img_data_list.append(img)
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255

print (img_data.shape)
fileObject = open(file_Name, 'wb')
pickle.dump(img_data, fileObject)

