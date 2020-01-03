import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd

## Destination for Numpy Files
np_path = 'Dataset/Image_Classification/Numpy Files'

## Read the CSV Data File
csv_path = 'Dataset/Image_Classification/dataset.csv'
df = pd.read_csv(csv_path)

x = df.iloc[:,0]
y = df.iloc[:,1]
x,y = shuffle(x,y)

## Encode the Class
le = LabelEncoder()
y = le.fit_transform(y)

## Split the files into train and test
x_train,x_val,y_train,y_val = train_test_split(x, y, test_size = 0.2)

# Read the Images
images_train = []
for i in x_train:
    img  = cv2.imread(i)
    images_train.append(img)

images_val = []
for i in x_val:
    img  = cv2.imread(i)
    images_val.append(img)

# np.save(np_path + '/' + 'x.npy', images)
# np.save(np_path + '/' + 'y.npy', y)
np.save(np_path + '/' + 'x_train.npy', np.array(images_train))
np.save(np_path + '/' + 'x_val.npy', np.array(images_val))
np.save(np_path + '/' + 'y_train.npy', y_train)
np.save(np_path + '/' + 'y_val.npy', y_val)
