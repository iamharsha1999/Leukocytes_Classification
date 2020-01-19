from Model import DL_Model
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
import numpy as np
from keras import backend as K
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import time
import seaborn as sn
import pandas as pd

class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return K.sigmoid(x) * x
get_custom_objects().update({'swish': Swish(swish)})

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    print("Plotting")
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


labels = ['Basophils', 'Eosinophils', 'Lymphocytes', 'Monocytes', 'Neutrophils']

## Weight path
weight_path = 'Image_Classification/Weights/SmallVGG_Fully_Swish/SGD_Momemtum/weights-94-0.97.hdf5'

## Load the Files
x_train = np.load('Image_Classification/Numpy Files/x_train.npy')
x_val   = np.load('Image_Classification/Numpy Files/x_val.npy')
y_train = np.load('Image_Classification/Numpy Files/y_train.npy')
y_val   = np.load('Image_Classification/Numpy Files/y_val.npy')
x       = np.load('Image_Classification/Numpy Files/x.npy')
y       = np.load('Image_Classification/Numpy Files/y.npy')
x = (x.astype(np.float32))/255
## Encode the labels

y_train = to_categorical(y_train, num_classes = 5)
y_val   = to_categorical(y_val, num_classes = 5)
y       = to_categorical(y, num_classes = 5)
y       = np.argmax(y,axis = 1)
model = DL_Model.build_feature_model(300, 300, 3, 5, 'smallvgg16')
model.load_weights(weight_path)

# yp_val = model.predict(x_val, verbose =1)
# yp_train = model.predict(x_train, verbose = 1)
start = time.process_time()
yp = model.predict(x, verbose = 1)
end = time.process_time()

yp = np.argmax(yp,axis=1)
# yp = to_categorical(yp, num_classes= 5)

accuracy = accuracy_score(yp, y)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(yp, y, average = 'macro')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(yp, y, average = 'macro')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(yp, y, average = 'macro')
print('F1 score: %f' % f1)

time_taken = end - start
print("Time Taken for Prediction: ", time_taken)

# confusion matrix
matrix = confusion_matrix(yp, y)
df_cm = pd.DataFrame(matrix, index = labels,columns = labels)
fig = plt.figure()
sn.heatmap(df_cm, annot=True)
fig.tight_layout()
plt.savefig('bla.png')
