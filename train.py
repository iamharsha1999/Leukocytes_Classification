
from Model import DL_Model
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras import backend as K
import  numpy as np

class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return K.sigmoid(x) * x

get_custom_objects().update({'swish': Swish(swish)})

x_train = np.load('Dataset/Image_Classification/Numpy Files/x_train.npy')
x_val = np.load('Dataset/Image_Classification/Numpy Files/x_val.npy')
y_train = np.load('Dataset/Image_Classification/Numpy Files/y_train.npy')
y_val = np.load('Dataset/Image_Classification/Numpy Files/y_val.npy')

y_train = to_categorical(y_train, num_classes = 5)
y_val   = to_categorical(y_val, num_classes = 5)

num_classes = 5

## Build the Model
model = DL_Model.build_feature_model(300, 300, 3, num_classes, 'smallvgg16')
model.compile(loss = 'categorical_crossentropy', optimizer  = 'adam', metrics = ['accuracy'])
model.summary()

## Create Checkpoints
weight_path = ''
weights = 'Dataset/Image_Classification/Weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoints = ModelCheckpoint(weights, verbose= 1, monitor= 'val_acc', save_best_only= True, mode = 'max')
callbacks = [checkpoints]

## Train the Model
epochs = 50
batch_size = 8

model.fit(x_train,y_train, verbose = 1, callbacks = callbacks, epochs = epochs, batch_size = batch_size, validation_data = (x_val, y_val))
