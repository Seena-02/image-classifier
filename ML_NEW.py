#!/usr/bin/env python
# coding: utf-8

# In[59]:


#import libraries
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[5]:


#load data
from keras.datasets import cifar10
#train data sets and test data sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[4]:


#check data types
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))


# In[9]:


#get shape of arrays
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('x_test shape:', y_test.shape)

#x_train 50,000 rows of data with 32x32 images with depth 3-> rgb


# In[24]:


#check first image as an array
#index changes the image
index = 10
x_train[index]


# In[25]:


#show above as image
img = plt.imshow(x_train[index])


# In[26]:


#get the image label
print('The image label is:', y_train[index])


# In[27]:


# create image class
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"  
]

#print image class
print('The image class is:', classes[y_train[index][0]])


# In[45]:


#convert labels into 10 numbers to input into neural network
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)


# In[30]:


print(y_train_one_hot)
#10 numbers "1" means the label that the image is


# In[31]:


#print new label of the current image
print("The one hot label is:", y_train_one_hot[index])


# In[32]:


#convert to values between 0 and 1
x_train = x_train / 255
x_test = x_test / 255


# In[33]:





# In[35]:


#create models architecture
model = Sequential()

#add first layer
model.add( Conv2D(32, (5,5), activation='relu', input_shape=(32,32,3)) )

#pooling layer, 2x2 pixel filter, gets max element from feature maps
model.add(MaxPooling2D(pool_size = (2,2)))

model.add( Conv2D(32, (5,5), activation='relu') )

#pooling layer #2
model.add(MaxPooling2D(pool_size = (2,2)))

#flattening layer to linear array
model.add(Flatten())

#add layer with 1000 neurons
model.add(Dense(1000, activation = 'relu'))

#add drop out layer
model.add(Dropout(0.5))

#add layer with 500 neurons
model.add(Dense(500, activation = 'relu'))

#add drop out layer
model.add(Dropout(0.5))

#add layer with 250 neurons
model.add(Dense(250, activation = 'relu'))

#add layer with 10 neurons
model.add(Dense(10, activation = 'softmax'))


# In[36]:


#Compile the model
model.compile(loss = "categorical_crossentropy",
             optimizer="adam",
             metrics=['accuracy'])


# In[37]:


#train our model
hist = model.fit(x_train, y_train_one_hot,
                batch_size = 256,
                epochs = 10,
                validation_split = 0.2)




# In[46]:


#evaluate model using test data
model.evaluate(x_test, y_test_one_hot)[1]


# In[51]:


#visualize model accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


# In[52]:


#visualize model's loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


# In[57]:


new_image = plt.imread('pancakes.jpg')
img = plt.imshow(new_image)


# In[62]:


#resize image
from skimage.transform import resize
resized_image = resize(new_image, (32,32,3))
img = plt.imshow(resized_image)


# In[63]:


#get model's predictions
predictions = model.predict(np.array([resized_image]))
predictions


# In[67]:


#sort the predictions from least to greatest
list_index = [0,1,2,3,4,5,6,7,8,9]
x = predictions

for i in range(10):
    for j in range(10):
        if x[0][list_index[i]] > x[0][list_index][j]:
            temp = list_index[i]
            list_index[i] = list_index[j]
            list_index[j] = temp
#show sorted labels
print(list_index)


# In[70]:


#print first 5 predictions(classes)
for i in range(10):
    print(classes[list_index[i]], ':', round(predictions[0][list_index][i] * 100,2), '%')


# In[ ]:





# %%
