import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

## Destination for Numpy Files
np_path = 'Dataset/Image_Classification/Numpy Files'

## Read the CSV Data File
csv_path = 'Dataset/Image_Classification/dataset.csv'
df = pd.read_csv(csv_path)

x = df.iloc[:,0]
y = df.iloc[:,1]
## Read the Images
images = []
for i in x:
    img  = cv2.imread(i)
    images.append(img)
## Encode the Class
le = LabelEncoder()
y = le.fit_transform(y)


## Split the files into train and test
x_train,x_val,y_train,y_val = train_test_split(images, y, test_size = 0.3)


np.save(np_path + '/' + 'x_train.npy', x_train)
np.save(np_path + '/' + 'x_val.npy', x_val)
np.save(np_path + '/' + 'y_train.npy', y_train)
np.save(np_path + '/' + 'y_val.npy', y_val)
