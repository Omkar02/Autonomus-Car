import pickle
import os
from PIL import Image
import cv2
import numpy as np
import numpy as array
print('*******************LOADING_DATA***********************************')
mylist = []
image_size=(120, 160)
file_Name = "training_image.pkl"
data= ('training')
print('*******************DATA_PRE-PROCESSING*************************************')
list = os.listdir(data) # dir is your directory path
number_files = len(list)
d = 0
# for files in os.listdir(data):
#     d += 1
#     outfile = 'resize_image/file_%d.jpg' % d
#     image=Image.open(os.path.join(data, files))
#     img = image.resize(image_size, Image.ANTIALIAS)
#     img.save(outfile, 'JPEG', quality=90)
#
#     percentage = (d / number_files) * 100
#     print('completed =',percentage)
#     if percentage >= 100:
#         print('Conversion Done!')
print('*******************PICKLING_STUFF***********************************')
def add_items(list,items):
    list.append(items)
    return list
print('*******************FINISHING_PICKLING***************************************')
there='resize_image'
list = os.listdir(there)
for file in os.listdir(there):
    img = cv2.imread(os.path.join(there, file), 0)
    img = img.flatten()
    print(img)
    mylist = add_items(mylist, img)
print('*******************LABLES***************************************')
label = np.ones((number_files), dtype=int)
label[0:number_files]=3
print('*******************Numpy array is DONE!!!!****************************************')
fileObject = open(file_Name, 'wb')
pickle.dump(mylist, fileObject)
print('*******************Pickling DONE!!!!*******************************************')

