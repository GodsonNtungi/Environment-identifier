# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 07:50:53 2022

@author: Godson Ntungi
"""

#new entry

#importing the required libraries
import pathlib
from tensorflow.keras.utils import image_dataset_from_directory
import tensorflow as tf
from tensorflow import keras
from keras import layers


#%%
#allocating the path of data
data= pathlib.Path('../seg_train/seg_train')
validation=pathlib.Path('../seg_test/seg_test')
test=pathlib.Path('../seg_pred')


#%%


#loading data

d_train=image_dataset_from_directory(data,
                                seed=123,
                                batch_size=32,
                                image_size=(128,128),
                                label_mode='categorical')

d_valid=image_dataset_from_directory(validation,
                                     seed=123,
                                     batch_size=32,
                                     image_size=(128,128),
                                     label_mode="categorical")

d_test=image_dataset_from_directory(test, 
                                    seed=123,
                                    batch_size=32,
                                    image_size=(128,128),)

classnames=d_train.class_names

#%%
#displaying some of the data
import matplotlib.pyplot as plt
for images,labels in d_train.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        #plt.title(classnames[labels[i]])
        plt.axis('off')
        
for images,labels in d_valid.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8')),
        #plt.title(classnames[labels[i]])
        plt.axis('off')

#%%


#performing data agumentation to reduce overfitting
data_agumentation=keras.Sequential([
    layers.RandomFlip('horizontal',input_shape=(128,128,3)),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2)])

#%%


#creating the model

model=keras.Sequential([
    data_agumentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(32,3,padding='same',activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(64,3,padding='same',activation='relu'),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(6,activation='softmax')])

#%%

model.compile(loss=['categorical_crossentropy'],optimizer='adam',metrics=['accuracy'])
model.build()
model.summary()

#%%


#training the model
with tf.device('/CPU:0'):
    history=model.fit(d_train,validation_data=d_valid,epochs=7,)

#%%


#saving the model
model.save('intelli1.h5')

#%%
#displaying accuracy and loss of the model
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']
epoch=range(7)

plt.subplot(1,2,1)
plt.plot(epoch,acc,label='accuracy')
plt.plot(epoch,val_acc,label='val_accuracy')
plt.title('accuracy')
plt.legend(loc='upper left')

plt.subplot(1,2,2)
plt.plot(epoch,loss,label='loss')
plt.plot(epoch,val_loss,label='val_loss')
plt.title('loss')
plt.legend(loc='upper left')




#%%

#loading the saved model

saved_model1=tf.keras.models.load_model('intelli1.h5')

#%%

#predicting testing data
with tf.device('/CPU:0'):
    predictions=saved_model1.predict(d_test)

#%%

print(predictions[5])























