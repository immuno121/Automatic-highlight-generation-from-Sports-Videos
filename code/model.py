from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, Conv2D, MaxPooling2D
from sklearn.metrics import roc_curve, auc,average_precision_score,precision_score,recall_score,precision_recall_curve
from sklearn.metrics import precision_recall_curve
from keras.optimizers import SGD, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils, generic_utils

import theano
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing
from keras import backend as K
from keras import optimizers
from keras import initializers

K.set_image_dim_ordering('th')
initializers.Initializer()

#np.set_printoptions(threshold='nan')

# image specification
img_rows, img_cols, img_depth = 256, 256,1

# Training data

X_tr = []  # variable to store entire train dataset
X_test = []  # variable to store entire test dataset

c = 0
frames = []

#listing = os.listdir('frames/046/046_c')
#listing = os.listdir('C:\\Users\\User\\Documents\\UMass Amherst\\Semester 1\\COMPSCI 670 - Computer Vision\\Project\\data_train\\data_train')
listing = os.listdir('C:\\Users\\User\\Documents\\UMass Amherst\\Semester 1\\COMPSCI 670 - Computer Vision\\Project\\frames\\frames\\046\\046_a_aug\\046_a')
#print (len(listing))
for i in range(len(listing)):

    #im = 'frames/046/046_c/frame%d.jpg' % (i)
    #im = 'C:\\Users\\User\\Documents\\UMass Amherst\\Semester 1\\COMPSCI 670 - Computer Vision\\Project\\TrainData\\train_c_frames3\\frame%d.jpg' % (i+1)
    im='C:\\Users\\User\\Documents\\UMass Amherst\\Semester 1\\COMPSCI 670 - Computer Vision\\Project\\frames\\frames\\046\\046_a_aug\\046_a\\frame%d.jpg' % (i)
    #im = 'C:\\Users\\User\\Documents\\UMass Amherst\\Semester 1\\COMPSCI 670 - Computer Vision\\Project\\data_train\\data_train\\frame%d.jpg' % (i+1)
    #print(im)
    #print(i)
    frame = cv2.imread(im,0)
    #print(frame.shape[0])
    #print(frame.shape[1])
    frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frames.append(frame)
    c += 1

    #if c % 30 == 0 and c != 0:
    inpt = np.array(frame)
    #frames = []
    #print (inpt.shape)

    #ipt = np.rollaxis(np.rollaxis(inpt, 2, 0), 2, 0)
    # print (ipt.shape)
    X_tr.append(inpt)
    #print(len(X_tr))
#############################################################################################

#listing = os.listdir('frames/frames/118/118_c')
#listing = os.listdir('C:\\Users\\User\\Documents\\UMass Amherst\\Semester 1\\COMPSCI 670 - Computer Vision\\Project\\TestData\\test_a_frames3')
#listing = os.listdir('C:\\Users\\User\\Documents\\UMass Amherst\\Semester 1\\COMPSCI 670 - Computer Vision\\Project\\data_test\\data_test')
listing=os.listdir('C:\\Users\\User\\Documents\\UMass Amherst\\Semester 1\\COMPSCI 670 - Computer Vision\\Project\\frames\\frames\\118\\118_a')
#listing=os.listdir('C:\\Users\\User\\Documents\\UMass Amherst\\Semester 1\\COMPSCI 670 - Computer Vision\\Project\\TrainData\\118\\118_a')
#print (len(listing))

for i in range(len(listing)):
    #im = 'frames/frames/118/118_c/frame%d.jpg' % (i)
    #im = 'C:\\Users\\User\\Documents\\UMass Amherst\\Semester 1\\COMPSCI 670 - Computer Vision\\Project\\TestData\\test_a_frames3\\frame%d.jpg' % (i+1)
    im='C:\\Users\\User\\Documents\\UMass Amherst\\Semester 1\\COMPSCI 670 - Computer Vision\\Project\\frames\\frames\\118\\118_a\\frame%d.jpg' % (i)
    #print(im)
    frame = cv2.imread(im,0)
    #print(frame.shape[0])
    #print(frame.shape[1])
    frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frames.append(frame)
    c += 1

    #if c % 30 == 0 and c != 0:
    inpt1 = np.array(frame)
    #frames = []
    #print (inpt1.shape)
    #ipt = np.rollaxis(np.rollaxis(inpt1, 2, 0), 2, 0)
    # print (ipt.shape)

    X_test.append(inpt1)
    #print(len(X_test))
#############################################################################################

X_tr_array = np.array(X_tr)  # convert the frames read into array
X_test_array = np.array(X_test)

num_samples = len(X_tr_array)
num_samples_test = len(X_test_array)

print("Number of samples =", num_samples)
print("Number of test_samples",num_samples_test)
# Assign Label to each class

label = np.zeros((num_samples,), dtype=int)
file = open('C:\\Users\\User\\Documents\\UMass Amherst\\Semester 1\\COMPSCI 670 - Computer Vision\\Project\\frames\\frames\\046\\046-Aug.txt')
#file = open('frames/046/046.txt')
#file = open('C:\\Users\\User\\Documents\\UMass Amherst\\Semester 1\\COMPSCI 670 - Computer Vision\\Project\\TrainData\\final_highlights.txt')
#file = open('C:\\Users\\User\\Documents\\UMass Amherst\\Semester 1\\COMPSCI 670 - Computer Vision\\Project\\TrainData\\final_highlights_aug.txt')
data = file.read()
#data = [data.strip() for ]
data=data.strip()
if data != '':
    data = data.split('\n')
    #data=data[0:len(data)-1:1]
    print("Train_highlights = ",data)
    #data=data
    for i in range(0, len(data)):
        x = int(data[i][0:2])
        for j in range(x*30, (x*30 + 10*30)):
            label[j-1] = 1
            #print(label.count(1))
print(np.count_nonzero(label))
label_test = np.zeros((num_samples_test,), dtype=int)
file = open('C:\\Users\\User\\Documents\\UMass Amherst\\Semester 1\\COMPSCI 670 - Computer Vision\\Project\\frames\\frames\\118\\118.txt')
#file = open('frames/frames/118/118.txt')
#file = open('C:\\Users\\User\\Documents\\UMass Amherst\\Semester 1\\COMPSCI 670 - Computer Vision\\Project\\TestData\\test_highlights.txt')
data = file.read()
data=data.strip()
print("Test Highlights = ",data)
if data != '':
    data = data.split('\n')
    #data = data[0:len(data) - 1:1]
    for i in range(0, len(data)):
        x = int(data[i][0:2])
        for j in range(x*30, (x*30 + 10*30)):
            label_test[j] = 1

train_data = [X_tr_array, label]
print ("Train_data",train_data)
test_data = [X_test_array, label_test]
print("Test_data",test_data)
(X_train, y_train) = (train_data[0], train_data[1])

(X_test, y_test) = (test_data[0], test_data[1])
print('X_Train shape:', X_train.shape)

train_set = np.zeros((num_samples, 1, img_rows, img_cols))
test_set = np.zeros((num_samples_test, 1, img_rows, img_cols))

for h in range(num_samples):
    train_set[h][0][:][:] = X_train[h, :, :]

for h in range(num_samples_test):
    test_set[h][0][:][:] = X_test[h, :, :]

print(train_set.shape, 'train samples')

# CNN Training parameters

batch_size = 2
nb_classes = 2
nb_epoch = 10
print("y_test 1",y_test)
print("Shape of y_train: {}".format(y_train.shape))
print("Shape of y_test: {}".format(y_test.shape))
#print("y_train.head()")
#print(y_train.head())
#print("y_test.head()")
#print(y_test.head())
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print("Shape of Y_train: {}".format(Y_train.shape))
print("Shape of Y_test: {}".format(Y_test.shape))
print("Number of positive training cases: {}".format(len(Y_train[Y_train == 1])))
print("Number of negative training cases: {}".format(len(Y_train[Y_train == 0])))
print("y_test 2",Y_test)
np.savetxt('Y_test.txt',Y_test,delimiter=',', newline='\n')
print("y_test 2",Y_test.shape)
# number of convolutional filters to use at each layer
nb_filters = [16, 16]
nb_filters2 = [10, 10]
nb_filters3 = [8, 8]
# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [2, 2]

# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [3, 3]
nb_conv2 = [2, 2]

# Pre-processing

train_set = train_set.astype('float32')

train_set -= np.mean(train_set)

train_set /= np.max(train_set)

test_set = test_set.astype('float32')

test_set -= np.mean(test_set)

test_set /= np.max(test_set)

# Define model

model = Sequential()
model.add(Conv2D(nb_filters[0], kernel_size=(nb_conv[0],nb_conv[0]),
                        input_shape=(1, img_rows, img_cols), activation='relu', kernel_initializer=initializers.he_uniform(seed=None), bias_initializer='zeros', padding = 'same'))
model.add(Conv2D(nb_filters[0], kernel_size=(nb_conv[0],nb_conv[0]),
                        input_shape=(1, img_rows, img_cols), activation='relu', kernel_initializer=initializers.he_uniform(seed=None), bias_initializer='zeros', padding = 'same'))
model.add(MaxPooling2D(pool_size=(nb_pool[0], nb_pool[0])))
#model.add(MaxPooling2D(pool_size=(nb_pool[0], nb_pool[0])))
model.add(Conv2D(nb_filters2[0], kernel_size=(nb_conv[0],nb_conv[0]),
                        input_shape=(1, img_rows, img_cols), activation='relu', kernel_initializer=initializers.he_uniform(seed=None), bias_initializer='zeros', padding = 'same'))
model.add(Conv2D(nb_filters2[0], kernel_size=(nb_conv[0],nb_conv[0]),
                        input_shape=(1, img_rows, img_cols), activation='relu', kernel_initializer=initializers.he_uniform(seed=None), bias_initializer='zeros', padding = 'same'))
model.add(MaxPooling2D(pool_size=(nb_pool[0], nb_pool[0])))
#model.add(MaxPooling2D(pool_size=(nb_pool[0], nb_pool[0])))
model.add(Conv2D(nb_filters3[0], kernel_size=(nb_conv[0],nb_conv[0]),
                        input_shape=(1, img_rows, img_cols, img_depth), activation='relu', kernel_initializer=initializers.he_uniform(seed=None), bias_initializer='zeros', padding = 'same'))
model.add(Conv2D(nb_filters3[0], kernel_size=(nb_conv[0],nb_conv[0]),
                        input_shape=(1, img_rows, img_cols, img_depth), activation='relu', kernel_initializer=initializers.he_uniform(seed=None), bias_initializer='zeros', padding = 'same'))
#model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))
#model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(200, activation='relu', kernel_initializer=initializers.he_uniform(seed=None), bias_initializer='zeros'))

#model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(80, activation='relu', kernel_initializer=initializers.he_uniform(seed=None), bias_initializer='zeros'))

#model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(nb_classes, init='normal'))

#model.add(BatchNormalization())

model.add(Activation('softmax'))

#sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='mean_squared_error', optimizer=sgd)
optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

# Split the data

X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(train_set, Y_train, test_size=0.2, random_state=4)

print (model.summary())

# Train the model

hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new, y_val_new),
                 batch_size=batch_size, nb_epoch=nb_epoch, shuffle=False)

# hist = model.fit(train_set, Y_train, batch_size=batch_size,
#         nb_epoch=nb_epoch,validation_split=0.2, show_accuracy=True,
#           shuffle=True)


# Evaluate the model
prediction = model.predict(test_set, batch_size=batch_size, verbose=0)
print("prediction",prediction)
np.savetxt('prediction.txt',prediction,delimiter=',', newline='\n')
print("prediction.shape[0]",prediction.shape[0])
y_score=[]
for i in range(prediction.shape[0]):
    if prediction[i,0] > prediction[i,1]:
        y_score.append(0)
    else:
        y_score.append(1)

score = model.evaluate(test_set, Y_test, batch_size=batch_size)
print('Test score:', score[0])
print('Test accuracy:', score[1])


#def generate_results(y_test, y_score):
fpr, tpr, _ = roc_curve(y_test,y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.savefig('ROC')
plt.show()
print('AUC: %f' % roc_auc)
average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
#print('Avg Precision score: %f' %  average_precision_score(y_test, y_score))
print('Precision score: %f' % precision_score(y_test, y_score, average='micro'))
print('Recall score: %f' % recall_score(y_test, y_score, average='weighted') )
precision, recall, _ = precision_recall_curve(y_test, y_score)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
plt.savefig('Precision-Recall')
