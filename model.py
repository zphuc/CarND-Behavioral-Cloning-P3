import os
import argparse
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

import matplotlib.pyplot as plt

############################################################
def getSamples(data_pathes):
    tsamples = []
    for dpath in data_pathes:
        with open(dpath + '/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)    # skip the first line
            for line in reader:
                 # update filepath of image file
                 line[0] = dpath + '/' + line[0]
                 line[1] = dpath + '/' + line[1]
                 line[2] = dpath + '/' + line[2]
                 tsamples.append(line)

    return tsamples        

############################################################
def generator(samples, batch_size=32):
    correction = 0.2 # 0.1, 0.3 #this is a parameter to tune
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0])
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                left_image = cv2.imread(batch_sample[1])
                images.append(left_image)
                angles.append(center_angle + correction)

                right_image = cv2.imread(batch_sample[2])
                images.append(right_image)
                angles.append(center_angle - correction)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

############################################################
###### ConvNet Definintion #################################
#-----------------------------------------------------------
def LeNetModel():
### LeNet model
    model = Sequential()

    # Preprocess incoming data, 
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,25),(0,0))))

    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model
#-----------------------------------------------------------
def NvidiaModel():
### Nvidia model
   
    model = Sequential()

    # Preprocess incoming data, 
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,25),(0,0))))

    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

#-----------------------------------------------------------
def NvidiaModelw4Drop():
### Nvidia model modified with Dropout
    model = Sequential()

    # Preprocess incoming data,
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,25),(0,0))))

    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Dropout(0.50))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.25))
    model.add(Dense(50))
    model.add(Dropout(0.25))
    model.add(Dense(10))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    return model

############################################################
def plotCropImg(img,simg):

    from keras.layers.core import K

    model = Sequential()
    #model.add(Lambda(lambda x: x, input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,25),(0,0))))

    output = K.function([model.layers[0].input], [model.layers[1].output])
    crop_img = output([img[None,...]])[0]

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.savefig(simg+"_org.png")

    plt.imshow(np.uint8(crop_img[0,...]), cmap='gray')
    plt.savefig(simg+"_crop.png")


############################################################

def doplotHis(hisObj):
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.xlim(0,5)
    plt.ylim(0,0.2)
    #plt.show()
    plt.savefig("pHis.png")

############################################################



############################################################
# Main Program
############################################################

if __name__ == '__main__':

   parser = argparse.ArgumentParser(description='Tranning Driving')
   parser.add_argument('-d', dest='data', type=str, default='udacity', help='Dataset:udacity,track1,track2,all')
   parser.add_argument('-m', dest='model', type=str, default='Nvidia', help='Network model:LeNet, Nvidia, Nvidiaw4Drop')
   parser.add_argument('-e', dest='nb_epoch', type=int , default=5, help='number of epoch')

   args = parser.parse_args()

   sdata=args.data
   if sdata == "udacity":
      data_pathes = ['../data_udacity']
   elif sdata == "track1":
      data_pathes = ['../data_udacity','../track1/run01','../track1/run02']
   elif sdata == "track2":
      data_pathes = ['../track2/run01','../track2/run02']
   else:
      data_pathes = ['../data_udacity','../track1/run01','../track1/run02','../track2/run01','../track2/run02']

   samples = getSamples(data_pathes)
   print(len(samples))

   #plotCropImg(cv2.imread(samples[0][0]), sdata+"_center")
   #plotCropImg(cv2.imread(samples[0][1]), sdata+"_left")
   #plotCropImg(cv2.imread(samples[0][2]), sdata+"_right")


   train_samples, validation_samples = train_test_split(samples, test_size=0.2)
   print(len(train_samples))
   print(len(validation_samples))

   # compile and train the model using the generator function
   train_generator      = generator(train_samples     , batch_size=32)
   validation_generator = generator(validation_samples, batch_size=32)


   smodel=args.model
   if smodel == 'LeNet':
       model = LeNetModel()
   elif smodel == 'Nvidia':
       model = NvidiaModel()
   elif smodel == 'Nvidiaw4Drop':
       model = NvidiaModelw4Drop()
   else:
       print("unknown")
       exit()

   model.compile(loss='mse', optimizer='adam')

   #model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
   history_object = model.fit_generator(train_generator, samples_per_epoch= \
                     3*len(train_samples), validation_data=validation_generator, \
                     nb_val_samples=3*len(validation_samples), nb_epoch=args.nb_epoch, verbose=1)

   doplotHis(history_object)

   model.save('model.h5')


   exit()
