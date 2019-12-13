#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyCon 2018:
Satellite data is for everyone: insights into modern remote sensing research
with open data and Python

"""
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16 as VGG
from keras.applications.densenet import DenseNet201 as DenseNet
from keras.optimizers import SGD
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from image_functions import preprocessing_image_rgb

# variables
path_to_split_datasets = "~/Documents/Data/PyCon/RGB"
use_vgg = True
batch_size = 64

# contruct path
path_to_home = os.path.expanduser("~")
path_to_split_datasets = path_to_split_datasets.replace("~", path_to_home)
path_to_train = os.path.join(path_to_split_datasets, "train")
path_to_validation = os.path.join(path_to_split_datasets, "validation")

# get number of classes
sub_dirs = [sub_dir for sub_dir in os.listdir(path_to_train)
            if os.path.isdir(os.path.join(path_to_train, sub_dir))]
num_classes = len(sub_dirs)

# parameters for CNN
if use_vgg:
    base_model = VGG(include_top=False,
                     weights='imagenet',
                     input_shape=(64, 64, 3))
else:
    base_model = DenseNet(include_top=False,
                          weights='imagenet',
                          input_shape=(64, 64, 3))
# add a global spatial average pooling layer
top_model = base_model.output
top_model = GlobalAveragePooling2D()(top_model)
# or just flatten the layers
#    top_model = Flatten()(top_model)
# let's add a fully-connected layer
if use_vgg:
    # only in VGG19 a fully connected nn is added for classfication
    # DenseNet tends to overfitting if using additionally dense layers
    top_model = Dense(2048, activation='relu')(top_model)
    top_model = Dense(2048, activation='relu')(top_model)
# and a logistic layer
predictions = Dense(num_classes, activation='softmax')(top_model)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# print network structure
model.summary()

# defining ImageDataGenerators
# ... initialization for training
train_datagen = ImageDataGenerator(fill_mode="reflect",
                                   rotation_range=45,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   preprocessing_function=preprocessing_image_rgb)
# ... initialization for validation
test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_image_rgb)
# ... definition for training
train_generator = train_datagen.flow_from_directory(path_to_train,
                                                    target_size=(64, 64),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
# just for information
class_indices = train_generator.class_indices
print(class_indices)

# ... definition for validation
validation_generator = test_datagen.flow_from_directory(path_to_validation,
                                                        target_size=(64, 64),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adadelta', loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# generate callback to save best model w.r.t val_categorical_accuracy
if use_vgg:
    file_name = "vgg"
else:
    file_name = "dense"

checkpointer = ModelCheckpoint("../data/models/" + file_name +
                               "_rgb_transfer_init." +
                               "{epoch:02d}-{val_categorical_accuracy:.3f}." +
                               "hdf5",
                               monitor='val_categorical_accuracy',
                               verbose=1,
                               save_best_only=True,
                               mode='max')

earlystopper = EarlyStopping(monitor='val_categorical_accuracy',
                             patience=10,
                             mode='max',
                             restore_best_weights=True)

tensorboard = TensorBoard(log_dir='./logs', write_graph=True, write_grads=True,
                          write_images=True, update_freq='epoch')

history = model.fit_generator(train_generator,
                              steps_per_epoch=1000,
                              epochs=10000,
                              callbacks=[checkpointer, earlystopper,
                                         tensorboard],
                              validation_data=validation_generator,
                              validation_steps=500)
initial_epoch = len(history.history['loss'])+1
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
names = []
for i, layer in enumerate(model.layers):
    names.append([i, layer.name, layer.trainable])
print(names)

if use_vgg:
    # we will freaze the first convolutional block and train all
    # remaining blocks, including top layers.
    for layer in model.layers[:4]:
        layer.trainable = False
    for layer in model.layers[4:]:
        layer.trainable = True
else:
    for layer in model.layers[:7]:
        layer.trainable = False
    for layer in model.layers[7:]:
        layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# generate callback to save best model w.r.t val_categorical_accuracy
if use_vgg:
    file_name = "vgg"
else:
    file_name = "dense"
checkpointer = ModelCheckpoint("../data/models/" + file_name +
                               "_rgb_transfer_final." +
                               "{epoch:02d}-{val_categorical_accuracy:.3f}" +
                               ".hdf5",
                               monitor='val_categorical_accuracy',
                               verbose=1,
                               save_best_only=True,
                               mode='max')
earlystopper = EarlyStopping(monitor='val_categorical_accuracy',
                             patience=50,
                             mode='max')
model.fit_generator(train_generator,
                    steps_per_epoch=1000,
                    epochs=10000,
                    callbacks=[checkpointer, earlystopper, tensorboard],
                    validation_data=validation_generator,
                    validation_steps=500,
                    initial_epoch=initial_epoch)
