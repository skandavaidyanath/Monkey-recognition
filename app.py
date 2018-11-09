# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 14:09:25 2018

@author: Skanda
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu', padding = 'same'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = 2))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = 2))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 10, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer= 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('validation',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch = 1097,
                         epochs = 15,
                         validation_data = test_set,
                         validation_steps = 272)

classifier.summary()
classifier.save('model.h5')
del classifier
classifier =  load_model('model.h5')

monkey_names = ['Mantled Howler', 'Patas monkey', 'Bald Uakari', 'Japanese Macaque', 'Pygmy Marmoset', 'White Headed Capuchin', 'Silvery Marmoset', 'Common Squirrel monkey', 'Black Headed Night monkey', 'Nilgiri Langur']
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('Bald-Uakari.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
classifier.predict(test_image)
result = classifier.predict(test_image)
index = np.where(result == 1)[1][0]
print(monkey_names[index])

