from Model import DL_Model, AdaBound
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras.optimizers import SGD
from keras import backend as K
import  numpy as np
import matplotlib.pyplot as plt


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

x_train = (x_train.astype(np.float32))/255
x_val = (x_val.astype(np.float32))/255

y_train = to_categorical(y_train, num_classes = 5)
y_val   = to_categorical(y_val, num_classes = 5)

num_classes = 5


# # AdaBound Optimizer
# optm = AdaBound(lr=1e-03, final_lr=0.1, gamma=1e-03, weight_decay=0., amsbound=False)


# Build the Model
## SGD Optmizer
sgd = SGD(lr = 1e-6, momentum = 0.9)
model = DL_Model.build_feature_model(300, 300, 3, num_classes, 'smallvgg16')
model.compile(loss = 'categorical_crossentropy', optimizer  = 'sgd', metrics = ['accuracy'])
model.summary()


# Train the Model
epochs = 100
batch_size = 16

# model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size),epochs = 100, steps_per_epoch = len(x_train)/batch_size, validation_data = (x_val,y_val))

# Create Checkpoints
weights = 'Dataset/Image_Classification/Weights/SmallVGG_Fully_Swish/weights-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoints = ModelCheckpoint(weights, verbose= 1, monitor= 'val_acc', save_best_only= True, mode = 'auto')
callbacks = [checkpoints]
history = model.fit(x_train,y_train, verbose = 1, callbacks = callbacks,epochs = epochs,batch_size = batch_size, validation_data = (x_val, y_val))



#Plotting the Model

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
