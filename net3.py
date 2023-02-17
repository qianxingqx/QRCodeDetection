# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 13:17:31 2023

@author: du
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# 读取15个文件夹中的图像，并将它们存储为numpy数组和对应的标签
def load_data(data_dir):
    images = []
    labels = []
    for i, folder in enumerate(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert('RGB')
            image = image.resize(IMAGE_SIZE)
            images.append(np.array(image))
            labels.append(i)
    return np.array(images), np.array(labels)


print("--Loading data...")
# 定义图像大小
IMAGE_SIZE = (512, 512)
# 加载图像和标签数据
data_dir = r"E:\Dataset\QRCodeDataset\TrainTestDataset\train"
images, labels = load_data(data_dir)

# 将标签进行one-hot编码
# labels = np.eye(15)[labels]
labels = np.eye(15)[labels][:, 0]

# 将数据分为训练集、验证集和测试集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# y_train = np.argmax(y_train, axis=1)
# y_val = np.argmax(y_val, axis=1)
print("--Data augmentation...")
# 定义数据增强器
datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    shear_range=0,
    zoom_range=0,
    horizontal_flip=False,
    fill_mode='nearest')

datagen_test = ImageDataGenerator(rescale=1./255)

print("--Construct model...")
# 基础模型是ResNet50，使用预训练权重
# base_model = ResNet50(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet')
# 预训练权重下载地址：https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
base_model = ResNet50(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='./resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

# 添加新的分类层和open set检测层
x = base_model.output
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(15, activation='softmax')(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=[x, output])

# 冻结ResNet50的前175层的权重
for layer in model.layers[:175]:
    layer.trainable = False

# 编译模型
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss=['categorical_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])

# 设置回调函数
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

print("--Training...")
# 训练模型
history = model.fit(
    datagen_train.flow(X_train, (y_train, np.zeros((y_train.shape[0], 1))), batch_size=32),
    steps_per_epoch=len(X_train) / 32,
    # validation_data=datagen_test.flow(X_val, (y_val.squeeze(), np.zeros((y_val.shape[0], 1))), batch_size=32),
    # validation_steps=len(X_val) / 32,
    epochs=50,
    callbacks=[checkpoint, early_stop]
)


# 绘制训练过程中的损失和准确率变化曲线
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.show()

print("--Evaluating...")
# 评估模型在测试集上的表现
score = model.evaluate(datagen_test.flow(X_test, [y_test, np.zeros(y_test.shape[0])], batch_size=32))
print('Test loss:', score[0])
print('Test accuracy:', score[3])
