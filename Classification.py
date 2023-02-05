import os
import csv
from PIL import Image
from numpy import asarray
import numpy as np
import cv2

from sklearn.model_selection import train_test_split 
from keras.utils.np_utils import to_categorical
from tensorflow.keras.applications import VGG16
from keras.models import Sequential,model_from_json
from keras.layers import Conv2D, Dense, Dropout, Flatten, Activation
from tensorflow.keras.metrics import BinaryAccuracy, Precision,Recall,AUC
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import tensorflow as tf
import keras.backend as K
import datetime

images_folder = os.path.join(os.getcwd(), "PatientData")
list_dir = [i for i in os.listdir(images_folder) if os.path.isdir(os.path.join(images_folder,i))]
image_list = []
label = []

def classify(i, file_name, classifier):
  img = cv2.imread(os.path.join(images_folder,i,file_name))
  img = cv2.resize(img, (256,256))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  image_list.append(img)
  label.append(1)

def Creating_label(images_folder, list_dir, classify):
    for i in list_dir:
      for file in os.listdir(os.path.join(images_folder,i)):
        if("thresh" in file ):
          file_2 = file.split("_")
          csv_file = csv.reader(open(os.path.join(images_folder,i+str("_Labels.csv")), "r"), delimiter=",")
          next(csv_file,None)
          for feild in csv_file:
            if(file_2[1] == feild[0]):
              if(int(feild[1])>0):
                classify(i, file, 1)
              else:
                classify(i, file, 0)

Creating_label(images_folder, list_dir, classify)
X = np.array(image_list)
Y = np.array(label)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=25)
model_id='base_model'
no_epochs = 100
early_stopping_patience = 50

base_model = VGG16(input_shape=(256,256, 3),include_top=False,weights="imagenet")
for layer in base_model.layers:
    layer.trainable=True

def creating_model(base_model):
    model=Sequential()
    model.add(base_model)
    model.add(Dropout(0.3))
    # Initializing layers with 8190 layer
    model.add(Dense(8190))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Dense(1020))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(130))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(20))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))
    return model

model = creating_model(base_model)


def changing_model(images_folder, X_train, X_test, Y_train, Y_test, model_id, no_epochs, early_stopping_patience, model):
    
    METRICS = [BinaryAccuracy(name='accuracy'),Precision(name='precision'),Recall(name='recall'),AUC(name='auc')]
    # reducing the model based on learning rate as 0.75 and patience as 5 with minimum learning rate 1e-10

    reducelr_plateau = ReduceLROnPlateau(monitor = 'val_loss',patience = 5,verbose = 1,factor = 0.75, min_lr = 1e-10)
    # Creating model checkpoint o check if model is being created
    model_checkpoint = ModelCheckpoint(filepath=images_folder + '/' + model_id + '.h5',save_freq='epoch',period=1)
    # Adding an early_stopping criteria as 50
    early_stopping = EarlyStopping(verbose=1, patience=early_stopping_patience)
    # Compiling the model with crossentropy category loss
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy',metrics=METRICS)
    # Fitting the training model
    history=model.fit(X_train, Y_train,validation_data=(X_test, Y_test),verbose = 1,epochs = no_epochs,callbacks=[reducelr_plateau,model_checkpoint,early_stopping])

    model.save(os.path.join(os.getcwd(), model_id+".h5"))

changing_model(images_folder, X_train, X_test, Y_train, Y_test, model_id, no_epochs, early_stopping_patience, model)