# -*- coding: utf-8 -*-
"""kfold.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uTabi4-pCIP0Rf4XhjnfN5A0Nk4pqaSb
"""





import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras_preprocessing import image
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import KMeansSMOTE
import smote_variants as sv
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from PIL import Image
from matplotlib import pyplot as plt

train_data = pd.read_csv('train.csv',dtype=str)
validation_data=pd.read_csv('test.csv',dtype=str)
Y = train_data[['label']]
X_train=train_data[['train_images']]
train_labels=list(train_data['label'])
kf = KFold(n_splits = 5)
Y_val=validation_data['label']

idg = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.3,
                         fill_mode='nearest',
                         horizontal_flip = True,
                         rescale=1./255)

VALIDATION_ACCURACY = []
accuracy_cnn=[]
f1_score_cnn=[]
sensitivity_cnn=[]
precision_cnn=[]
precision=[]
VALIDAITON_LOSS = []
f1_score=[]
recall=[]
image_dir_train='/content/CovidDataset/Train'
image_dir_val='/content/CovidDataset/Val'
save_dir = '/saved_models/'
fold_var = 1

for train_index, val_index in kf.split(np.zeros(224),Y):
    
    train_data_generator = idg.flow_from_dataframe(train_data, directory = image_dir_train,
                    x_col = "train_images", y_col = "label", target_size=(224, 224),
                    class_mode = "binary", shuffle = True)
    valid_data_generator  = idg.flow_from_dataframe(validation_data, directory = image_dir_val,
                x_col = "val_images", y_col = "label", target_size=(224, 224),
                class_mode = "binary", shuffle = True)
    sm = SMOTE(random_state=2)
    x_train,y_train= sm.fit_resample(train_data_generator,Y)

    
    # CREATE NEW MODEL
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss= keras.losses.binary_crossentropy,optimizer='adam', metrics=['accuracy'])
    
    history = model.fit(x_train,y_train,
            epochs=100,           
            validation_data=valid_data_generator)
    

    VALIDATION_ACCURACY.append(results['accuracy'])
    VALIDATION_LOSS.append(results['loss'])
    accuracy_cnn.append(results['accuracy'])
    tf.keras.backend.clear_session()
    test_y=[]
    train_y=[]
    for i in os.listdir("CovidDataset/Val/Normal/"):
      img=image.load_img("CovidDataset/Val/Normal/"+i)
      img=img.resize((224,224), Image.ANTIALIAS)
      img=image.img_to_array(img)
      img=np.expand_dims(img, axis=0)
      p=model.predict_classes(img)
      test_y.append(p[0,0])
      train_y.append(1)
    for i in os.listdir("CovidDataset/Val/Covid/"):
      img=image.load_img("CovidDataset/Val/Covid/"+i)
      img=img.resize((224,224), Image.ANTIALIAS)
      img=image.img_to_array(img)
      img=np.expand_dims(img, axis=0)
      p=model.predict_classes(img)
      test_y.append(p[0,0])
      train_y.append(0)
    train_y=np.array(train_y)
    test_y=np.array(test_y)
    f1_score=f1_score(train_y,test_y)
    f1_score_cnn.append(f1_score)
    recall=recall_score(train_y,test_y)
    sensitivity_cnn.append(recall)
    precision=precision_score(train_y,test_y)
    precision_cnn.append(precision)
    fold_var += 1

acc_avg_cnn=np.mean(accuracy_cnn)
acc_std_cnn=np.std(accuracy_cnn)

prec_avg_cnn=np.mean(precision_cnn)
prec_std_cnn=np.std(precision_cnn)

sens_avg_cnn=np.mean(sensitivity_cnn)
sens_std_cnn=np.std(sensitivity_cnn)

f1_avg_cnn=np.mean(f1_score_cnn)
f1_acc_cnn=np.std(f1_score_cnn)



VALIDATION_ACCURACY = []
accuracy_cnn_adasyn=[]
f1_score_cnn_adasyn=[]
sensitivity_cnn_adasyn=[]
precision_cnn_adasyn=[]
precision=[]
VALIDAITON_LOSS = []
f1_score=[]
recall=[]
image_dir_train='/content/CovidDataset/Train'
image_dir_val='/content/CovidDataset/Val'
save_dir = '/saved_models/'
fold_var = 1

for train_index, val_index in kf.split(np.zeros(224),Y):
    
    train_data_generator = idg.flow_from_dataframe(train_data, directory = image_dir_train,
                    x_col = "train_images", y_col = "label", target_size=(224, 224),
                    class_mode = "binary", shuffle = True)
    valid_data_generator  = idg.flow_from_dataframe(validation_data, directory = image_dir_val,
                x_col = "val_images", y_col = "label", target_size=(224, 224),
                class_mode = "binary", shuffle = True)
    sm = ADASYN(random_state=2)
    x_train,y_train= sm.fit_resample(train_data_generator,Y)

    
    # CREATE NEW MODEL
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss= keras.losses.binary_crossentropy,optimizer='adam', metrics=['accuracy'])
    
    history = model.fit(x_train,y_train,
            epochs=100,           
            validation_data=valid_data_generator)
   
    

    VALIDATION_ACCURACY.append(results['accuracy'])
    VALIDATION_LOSS.append(results['loss'])
    accuracy_cnn_adasyn.append(results['accuracy'])
    tf.keras.backend.clear_session()
    test_y=[]
    train_y=[]
    for i in os.listdir("CovidDataset/Val/Normal/"):
      img=image.load_img("CovidDataset/Val/Normal/"+i)
      img=img.resize((224,224), Image.ANTIALIAS)
      img=image.img_to_array(img)
      img=np.expand_dims(img, axis=0)
      p=model.predict_classes(img)
      test_y.append(p[0,0])
      train_y.append(1)
    for i in os.listdir("CovidDataset/Val/Covid/"):
      img=image.load_img("CovidDataset/Val/Covid/"+i)
      img=img.resize((224,224), Image.ANTIALIAS)
      img=image.img_to_array(img)
      img=np.expand_dims(img, axis=0)
      p=model.predict_classes(img)
      test_y.append(p[0,0])
      train_y.append(0)
    train_y=np.array(train_y)
    test_y=np.array(test_y)
    f1_score=f1_score(train_y,test_y)
    f1_score_cnn_adasyn.append(f1_score)
    recall=recall_score(train_y,test_y)
    sensitivity_cnn_adasyn.append(recall)
    precision=precision_score(train_y,test_y)
    precision_cnn_adasyn.append(precision)
    fold_var += 1

acc_avg_cnn_adasyn=np.mean(accuracy_cnn_adasyn)
acc_std_cnn_adasyn=np.std(accuracy_cnn_adasyn)
prec_avg_cnn_adasyn=np.mean(precision_cnn_adasyn)
prec_std_cnn_adasyn=np.std(precision_cnn_adasyn)
sens_avg_cnn_adasyn=np.mean(sensitivity_cnn_adasyn)
sens_std_cnn_adasyn=np.std(sensitivity_cnn_adasyn)
f1_avg_cnn_adasyn=np.mean(f1_score_cnn_adasyn)
f1_acc_cnn_adasyn=np.std(f1_score_cnn_adasyn)



oversampler= smote_variants.CURE_SMOTE()
X_samp, y_samp= oversampler.sample(X, y)

VALIDATION_ACCURACY = []
accuracy_cnn_border=[]
f1_score_cnn_border=[]
sensitivity_cnn_border=[]
precision_cnn_border=[]
precision=[]
VALIDAITON_LOSS = []
f1_score=[]
recall=[]
image_dir_train='/content/CovidDataset/Train'
image_dir_val='/content/CovidDataset/Val'
save_dir = '/saved_models/'
fold_var = 1

for train_index, val_index in kf.split(np.zeros(224),Y):
    
    train_data_generator = idg.flow_from_dataframe(train_data, directory = image_dir_train,
                    x_col = "train_images", y_col = "label", target_size=(224, 224),
                    class_mode = "binary", shuffle = True)
    valid_data_generator  = idg.flow_from_dataframe(validation_data, directory = image_dir_val,
                x_col = "val_images", y_col = "label", target_size=(224, 224),
                class_mode = "binary", shuffle = True)
    sm = BorderlineSMOTE(random_state=2)
    x_train,y_train= sm.fit_resample(train_data_generator,Y)

    
    # USE the NEW MODEL
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss= keras.losses.binary_crossentropy,optimizer='adam', metrics=['accuracy'])
    
    history = model.fit(x_train,y_train,
            epochs=100,           
            validation_data=valid_data_generator)
    

    VALIDATION_ACCURACY.append(results['accuracy'])
    VALIDATION_LOSS.append(results['loss'])
    accuracy_cnn_border.append(results['accuracy'])
    tf.keras.backend.clear_session()
    test_y=[]
    train_y=[]
    for i in os.listdir("CovidDataset/Val/Normal/"):
      img=image.load_img("CovidDataset/Val/Normal/"+i)
      img=img.resize((224,224), Image.ANTIALIAS)
      img=image.img_to_array(img)
      img=np.expand_dims(img, axis=0)
      p=model.predict_classes(img)
      test_y.append(p[0,0])
      train_y.append(1)
    for i in os.listdir("CovidDataset/Val/Covid/"):
      img=image.load_img("CovidDataset/Val/Covid/"+i)
      img=img.resize((224,224), Image.ANTIALIAS)
      img=image.img_to_array(img)
      img=np.expand_dims(img, axis=0)
      p=model.predict_classes(img)
      test_y.append(p[0,0])
      train_y.append(0)
    train_y=np.array(train_y)
    test_y=np.array(test_y)
    f1_score=f1_score(train_y,test_y)
    f1_score_cnn_border.append(f1_score)
    recall=recall_score(train_y,test_y)
    sensitivity_cnn_border.append(recall)
    precision=precision_score(train_y,test_y)
    precision_cnn_border.append(precision)
    fold_var += 1

acc_avg_cnn_border=np.mean(accuracy_cnn_border)
acc_std_cnn_border=np.std(accuracy_cnn_border)
prec_avg_cnn_border=np.mean(precision_cnn_border)
prec_std_cnn_border=np.std(precision_cnn_border)
sens_avg_cnn_border=np.mean(sensitivity_cnn_border)
sens_std_cnn_border=np.std(sensitivity_cnn_border)
f1_avg_cnn_border=np.mean(f1_score_cnn_border)
f1_acc_cnn_border=np.std(f1_score_cnn_border)



VALIDATION_ACCURACY = []
accuracy_cnn_kmeans=[]
f1_score_cnn_kmeans=[]
sensitivity_cnn_kmeans=[]
precision_cnn_kmeans=[]
precision=[]
VALIDAITON_LOSS = []
f1_score=[]
recall=[]
image_dir_train='/content/CovidDataset/Train'
image_dir_val='/content/CovidDataset/Val'
save_dir = '/saved_models/'
fold_var = 1

for train_index, val_index in kf.split(np.zeros(224),Y):
    
    train_data_generator = idg.flow_from_dataframe(train_data, directory = image_dir_train,
                    x_col = "train_images", y_col = "label", target_size=(224, 224),
                    class_mode = "binary", shuffle = True)
    valid_data_generator  = idg.flow_from_dataframe(validation_data, directory = image_dir_val,
                x_col = "val_images", y_col = "label", target_size=(224, 224),
                class_mode = "binary", shuffle = True)
    sm = KMeansSMOTE(random_state=2)
    x_train,y_train= sm.fit_resample(train_data_generator,Y)

    
    # CREATE NEW MODEL
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss= keras.losses.binary_crossentropy,optimizer='adam', metrics=['accuracy'])
    odel
    # FIT THE MODEL
    history = model.fit(x_train,y_train,
            epochs=100,           
            validation_data=valid_data_generator)
   

    VALIDATION_ACCURACY.append(results['accuracy'])
    VALIDATION_LOSS.append(results['loss'])
    accuracy_cnn_kmeans.append(results['accuracy'])
    tf.keras.backend.clear_session()
    test_y=[]
    train_y=[]
    for i in os.listdir("CovidDataset/Val/Normal/"):
      img=image.load_img("CovidDataset/Val/Normal/"+i)
      img=img.resize((224,224), Image.ANTIALIAS)
      img=image.img_to_array(img)
      img=np.expand_dims(img, axis=0)
      p=model.predict_classes(img)
      test_y.append(p[0,0])
      train_y.append(1)
    for i in os.listdir("CovidDataset/Val/Covid/"):
      img=image.load_img("CovidDataset/Val/Covid/"+i)
      img=img.resize((224,224), Image.ANTIALIAS)
      img=image.img_to_array(img)
      img=np.expand_dims(img, axis=0)
      p=model.predict_classes(img)
      test_y.append(p[0,0])
      train_y.append(0)
    train_y=np.array(train_y)
    test_y=np.array(test_y)
    f1_score=f1_score(train_y,test_y)
    f1_score_cnn_kmeans.append(f1_score)
    recall=recall_score(train_y,test_y)
    sensitivity_cnn_kmeans.append(recall)
    precision=precision_score(train_y,test_y)
    precision_cnn_kmeans.append(precision)
    fold_var += 1

acc_avg_cnn_kmeans=np.mean(accuracy_cnn_kmeans)
acc_std_cnn_kmeans=np.std(accuracy_cnn_kmeans)
prec_avg_cnn_kmeans=np.mean(precision_cnn_kmeans)
prec_std_cnn_kmeans=np.std(precision_cnn_kmeans)
sens_avg_cnn_kmeans=np.mean(sensitivity_cnn_kmeans)
sens_std_cnn_kmeans=np.std(sensitivity_cnn_kmeans)
f1_avg_cnn_kmeans=np.mean(f1_score_cnn_kmeans)
f1_acc_cnn_kmeans=np.std(f1_score_cnn_kmeans)



VALIDATION_ACCURACY = []
accuracy_cnn_cure=[]
f1_score_cnn_cure=[]
sensitivity_cnn_cure=[]
precision_cnn_cure=[]
precision=[]
VALIDAITON_LOSS = []
f1_score=[]
recall=[]
image_dir_train='/content/CovidDataset/Train'
image_dir_val='/content/CovidDataset/Val'
save_dir = '/saved_models/'
fold_var = 1

for train_index, val_index in kf.split(np.zeros(224),Y):
    
    train_data_generator = idg.flow_from_dataframe(train_data, directory = image_dir_train,
                    x_col = "train_images", y_col = "label", target_size=(224, 224),
                    class_mode = "binary", shuffle = True)
    valid_data_generator  = idg.flow_from_dataframe(validation_data, directory = image_dir_val,
                x_col = "val_images", y_col = "label", target_size=(224, 224),
                class_mode = "binary", shuffle = True)
    oversampler= sv.CURE_SMOTE()
    x_train,y_train= oversampler.sample(train_data_generator,Y)

    
    # CREATE NEW MODEL
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss= keras.losses.binary_crossentropy,optimizer='adam', metrics=['accuracy'])
   
    history = model.fit(x_train,y_train,
            epochs=100,           
            validation_data=valid_data_generator)
    
    

    VALIDATION_ACCURACY.append(results['accuracy'])
    VALIDATION_LOSS.append(results['loss'])
    accuracy_cnn_cure.append(results['accuracy'])
    tf.keras.backend.clear_session()
    test_y=[]
    train_y=[]
    for i in os.listdir("CovidDataset/Val/Normal/"):
      img=image.load_img("CovidDataset/Val/Normal/"+i)
      img=img.resize((224,224), Image.ANTIALIAS)
      img=image.img_to_array(img)
      img=np.expand_dims(img, axis=0)
      p=model.predict_classes(img)
      test_y.append(p[0,0])
      train_y.append(1)
    for i in os.listdir("CovidDataset/Val/Covid/"):
      img=image.load_img("CovidDataset/Val/Covid/"+i)
      img=img.resize((224,224), Image.ANTIALIAS)
      img=image.img_to_array(img)
      img=np.expand_dims(img, axis=0)
      p=model.predict_classes(img)
      test_y.append(p[0,0])
      train_y.append(0)
    train_y=np.array(train_y)
    test_y=np.array(test_y)
    f1_score=f1_score(train_y,test_y)
    f1_score_cnn_cure.append(f1_score)
    recall=recall_score(train_y,test_y)
    sensitivity_cnn_cure.append(recall)
    precision=precision_score(train_y,test_y)
    precision_cnn_cure.append(precision)
    fold_var += 1

acc_avg_cnn_cure=np.mean(accuracy_cnn_cure)
acc_std_cnn_cure=np.std(accuracy_cnn_cure)
prec_avg_cnn_cure=np.mean(precision_cnn_cure)
prec_std_cnn_cure=np.std(precision_cnn_cure)
sens_avg_cnn_cure=np.mean(sensitivity_cnn_cure)
sens_std_cnn_cure=np.std(sensitivity_cnn_cure)
f1_avg_cnn_cure=np.mean(f1_score_cnn_cure)
f1_acc_cnn_cure=np.std(f1_score_cnn_cure)



VALIDATION_ACCURACY = []
accuracy_cnn_normal=[]
f1_score_cnn_normal=[]
sensitivity_cnn_normal=[]
precision_cnn_normal=[]
precision=[]
VALIDAITON_LOSS = []
f1_score=[]
recall=[]
image_dir_train='/content/CovidDataset/Train'
image_dir_val='/content/CovidDataset/Val'
save_dir = '/saved_models/'
fold_var = 1

for train_index, val_index in kf.split(np.zeros(224),Y):
    
    train_data_generator = idg.flow_from_dataframe(train_data, directory = image_dir_train,
                    x_col = "train_images", y_col = "label", target_size=(224, 224),
                    class_mode = "binary", shuffle = True)
    valid_data_generator  = idg.flow_from_dataframe(validation_data, directory = image_dir_val,
                x_col = "val_images", y_col = "label", target_size=(224, 224),
                class_mode = "binary", shuffle = True)

    # CREATE NEW MODEL
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss= keras.losses.binary_crossentropy,optimizer='adam', metrics=['accuracy'])

    history = model.fit(train_data_generator,
            epochs=100,           
            validation_data=valid_data_generator)
 
    VALIDATION_ACCURACY.append(results['accuracy'])
    VALIDATION_LOSS.append(results['loss'])
    accuracy_cnn_normal.append(results['accuracy'])
    tf.keras.backend.clear_session()
    test_y=[]
    train_y=[]
    for i in os.listdir("CovidDataset/Val/Normal/"):
      img=image.load_img("CovidDataset/Val/Normal/"+i)
      img=img.resize((224,224), Image.ANTIALIAS)
      img=image.img_to_array(img)
      img=np.expand_dims(img, axis=0)
      p=model.predict_classes(img)
      test_y.append(p[0,0])
      train_y.append(1)
    for i in os.listdir("CovidDataset/Val/Covid/"):
      img=image.load_img("CovidDataset/Val/Covid/"+i)
      img=img.resize((224,224), Image.ANTIALIAS)
      img=image.img_to_array(img)
      img=np.expand_dims(img, axis=0)
      p=model.predict_classes(img)
      test_y.append(p[0,0])
      train_y.append(0)
    train_y=np.array(train_y)
    test_y=np.array(test_y)
    f1_score=f1_score(train_y,test_y)
    f1_score_cnn_normal.append(f1_score)
    recall=recall_score(train_y,test_y)
    sensitivity_cnn_normal.append(recall)
    precision=precision_score(train_y,test_y)
    precision_cnn_normal.append(precision)
    fold_var += 1

acc_avg_cnn_normal=np.mean(accuracy_cnn_normal)
acc_std_cnn_normal=np.std(accuracy_cnn_normal)
prec_avg_cnn_normal=np.mean(precision_cnn_normal)
prec_std_cnn_normal=np.std(precision_cnn_normal)
sens_avg_cnn_normal=np.mean(sensitivity_cnn_normal)
sens_std_cnn_normal=np.std(sensitivity_cnn_normal)
f1_avg_cnn_normal=np.mean(f1_score_cnn_normal)
f1_acc_cnn_normal=np.std(f1_score_cnn_normal)



