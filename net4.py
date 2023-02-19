# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:18:13 2023

@author: du
"""
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

# 定义图像的大小
IMG_SIZE = (512, 512)

# 定义训练集和测试集的比例
TRAIN_RATIO = 0.8
TEST_RATIO = 1 - TRAIN_RATIO

# 定义训练集和测试集的路径
DATA_PATH = r"E:\Dataset\QRCodeDataset\TrainTestDataset"
TRAIN_PATH = os.path.join(DATA_PATH, "train/")
TEST_PATH = os.path.join(DATA_PATH, "test/")

# 获取所有的分类标签
labels = sorted(os.listdir(TRAIN_PATH))

# 定义数据增强器
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# 读取数据并进行数据增强
train_generator = datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical')

test_generator = datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical')

# 构建模型
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(len(labels), activation='softmax'))

# 编译模型
model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size)

# 测试模型
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print('Test accuracy:', test_acc)
