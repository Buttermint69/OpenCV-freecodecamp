# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
!pip install caer canaro

# %%
import os
import caer
import numpy as np
import cv2 as cv
import gc
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
import canaro as cn

# %%
image_size = (80,80)
channels = 1
charac_loc = "../input/the-simpsons-characters-dataset/simpsons_dataset"

# %%
char_dict = dict()
for char in os.listdir(charac_loc):
    if char == "simpsons_dataset":
        pass
    else:
        char_dict[char] = len(os.path.join(charac_loc, char))

# Sorting char_dict
char_dict = caer.sort_dict(char_dict, descending =True)
char_dict

# %%
characters = [char_dict[i][0] for i in range(0,11)]
characters

# %%
# Training Data
train = caer.preprocess_from_dir(charac_loc, characters,
                                 channels = channels, IMG_SIZE= image_size, 
                                  isShuffle= True)

# %%
len(train)

# %%
plt.figure(figsize=(80,80))
plt.imshow(train[0][0], cmap="gray")
plt.show()

# %%
train_img, label = caer.sep_train(train, IMG_SIZE=image_size)

# %%
# Normalizing features ---> (0,1)
Norm_train_img = caer.normalize(train_img)
labels = to_categorical(label, num_classes= len(characters))

# %%
# Spliting data between train and validation set
x_train, x_val, y_train, y_val = caer.train_val_split(Norm_train_img, labels, val_ratio= 0.1)

# %%
del train
del label
del train_img
gc.collect()

# %%
BATCH_SIZE = 64
EPOCHS = 50

# %%
# Image data generator
datagen = cn.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size =BATCH_SIZE)

# %%
# MODEL
model = cn.models.createSimpsonsModel(IMG_SIZE=image_size, channels= channels,
                                     output_dim= len(characters), loss= "binary_crossentropy",
                                     decay = 1e-6, learning_rate=0.001, momentum=0.9, nesterov= True)

# %%
model.summary()

# %%
callbacks = [LearningRateScheduler(cn.lr_schedule)]

# %%
training = model.fit(train_gen, steps_per_epoch= len(x_train)//BATCH_SIZE,
                    epochs = EPOCHS, validation_data= (x_val, y_val),
                    validation_steps= len(y_val)//BATCH_SIZE, callbacks=callbacks)

# %%
"""
# Testing
"""

# %%
test_path = r'../input/the-simpsons-characters-dataset/kaggle_simpson_testset/kaggle_simpson_testset/charles_montgomery_burns_0.jpg'

img = cv.imread(test_path)

plt.imshow(img)
plt.show()


# %%
def prepare(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, image_size)
    image = caer.reshape(image, image_size, 1)
    return image

predictions = model.predict(prepare(img))
# Getting class with the highest probability
print(characters[np.argmax(predictions[0])])