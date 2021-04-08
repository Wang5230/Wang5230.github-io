---
title: TensorFlow2.x学习笔记
date: 2020-08-26 10:43:57
tags: 
- python
- machinelearning
---

## 基础

创建变量等操作与1.x基本类似，但可以进行即时执行模式，与之前的图执行模式不同，语法更精炼简单。

``` python
num_epoch = 100000
optimizer = tf.keras.optimizers.SGD(learning_rate = 5e-4)
for e in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = a * x + b
        loss = tf.reduce_sum(tf.square(y - y_pred))
    grads = tape.gradient(loss , variables)
    optimizer.apply_gradients(grads_and_vars = zip(grads , variables))
```

可以直接调用GradientTape来进行梯度以及导数计算。上面的代码中声明了一个梯度下降优化器，可以通过计算的求导结果更新模型参数，从而最小化某个特定的损失函数，通过apply_gradients()方法来进行调用，函数内的参数为需要更新的变量以及损失函数关于该变量的偏导数。需要传入一个列表，每个元素为(偏导，变量)。

## 模型与层

使用TF内置的库Keras来进行模型的构建。
Keras里有两个重要的概念：**层(Layer)**和**模型(model)**,层将各种计算流程和变量进行了封装(全连接层，卷积层，池化层等)，模型则用于各种层进行连接。模型通过类的形式呈现，所以可以通过继承tf.keras.Model来定义自己的模型，需要重写__init__()和call(),继承模型后可以调用模型中的方法和属性。可以大量简化代码,下面是通过模型的方式编写的一个简单的线性回归，可以看出在进行计算处理部分与之前基本一致，但是关于模型变量的访问，以及模型初始化都方便了许多。通过在模型内部实例化了一个全连接层，并对其在call()中进行调用

``` python
X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])

class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units = 1,
            activation=None,
            kernel_initializer = tf.zeros_initializer(),
            bias_initializer = tf.zeros_initializer()
        )
    def call(self , input):
        output = self.dense(input)
        return output
model = Linear()
optimizer = tf.keras.optimizers.SGD(lr=0.01)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.reduce_mean(tf.square(y_pred - y))
    grads = tape.gradient(loss , model.variables)
    optimizer.apply_gradients(grads_and_vars = zip(grads , model.variables))
print(model.variables)
```

**全连接层(Dense)** 是最常用的层之一，对输入矩阵进行f(AW+b)的线性变换+激活函数操作，激活函数一般是Relu等等.包含的参数如下

- units 输出张量的维度
- activation 激活函数 不指定时为f(x) = x
- use_bias 添加偏置，默认为True
- kernel_initializer,bias_initializer 权重，偏置的初始化器，默认为glort_uniform_initializer(很多层都默认使用)，使用ezeros_initializer表示初始化为0
  
**softmax函数**为了使得模型的输出能始终满足这两个条件，我们使用 Softmax 函数 （归一化指数函数， tf.nn.softmax ）对模型的原始输出进行归一化 。不仅如此，softmax 函数能够凸显原始向量中最大的值，并抑制远低于最大值的其他分量，这也是该函数被称作 softmax 函数的原因（即平滑化的 argmax 函数）。

### 模型的训练

tf.keras.losses 和 tf.keras.optimizer
需要定义一些模型超参数

- num_epochs = 5
- batch_size = 50
- learning_rate = 0.001
之后实例化模型和优化器，迭代进行数据的读取以及模型的训练步骤如下

- 从 DataLoader 中随机取一批训练数据；
- 将这批数据送入模型，计算出模型的预测值；
- 将模型预测值与真实值进行比较，计算损失函数（loss）。这里使用 tf.keras.losses 中的交叉熵函数作为损失函数
- 计算损失函数关于模型变量的导数；

- 将求出的导数值传入优化器，使用优化器的 apply_gradients 方法更新模型参数以最小化损失函数

### 模型的评估

使用tf.keras.metrics中的SparseCategoricalAccuracy来评估模型在测试集上的性能，通过将预测结果与真是结果相比较，输出正确的占比。通过update_state()方法向评估器输入预测值和真实值。其内部有变量来保存当前评估的指标的数值，最终通过result()方法输出最终评估值。

### 代码汇总

``` python
import numpy as np
import tensorflow as tf
#CNN with keras
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,             # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            activation=tf.nn.relu   # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)                  # [batch_size, 28, 28, 32]
        x = self.pool1(x)                       # [batch_size, 14, 14, 32]
        x = self.conv2(x)                       # [batch_size, 14, 14, 64]
        x = self.pool2(x)                       # [batch_size, 7, 7, 64]
        x = self.flatten(x)                     # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)                      # [batch_size, 1024]
        x = self.dense2(x)                      # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output
# MNISTLoader and MLP
class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)      # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)        # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]




# class MLP(tf.keras.Model):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._flatten = tf.keras.layers.Flatten()
#         self.dense1 = tf.keras.layers.Dense(units=100 , activation = tf.nn.relu)
#         self.dense2 = tf.keras.layers.Dense(units=10)
#     def call(self, inputs):
#         x = self._flatten(inputs)
#         x = self.dense1(x)
#         x = self.dense2(x)
#         output = tf.nn.softmax(x)
#         return output
num_epochs = 5
batch_size = 50
learning_rate = 0.001
model = CNN()
data_loader = MNISTLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
for batch_index in range(num_batches):
    X , y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y , y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f"%(batch_index , loss.numpy()))
    grads = tape.gradient(loss , model.variables)
    optimizer.apply_gradients(grads_and_vars = zip(grads , model.variables))
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(data_loader.num_test_data // batch_size)
for batch_index in range(num_batches):
    start_index , end_index = batch_index * batch_size , (batch_index + 1) * batch_size
    y_pred = model.predict(data_loader.test_data[start_index : end_index])
    sparse_categorical_accuracy.update_state(y_true = data_loader.test_label[start_index : end_index] , y_pred= y_pred)
print("test accuracy : %f"%sparse_categorical_accuracy.result())
#using Model for linear regression
# X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# y = tf.constant([[10.0], [20.0]])

# class Linear(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.dense = tf.keras.layers.Dense(
#             units = 1,
#             activation=None,
#             kernel_initializer = tf.zeros_initializer(),
#             bias_initializer = tf.zeros_initializer()
#         )
#     def call(self , input):
#         output = self.dense(input)
#         return output
# model = Linear()
# optimizer = tf.keras.optimizers.SGD(lr=0.01)
# for i in range(100):
#     with tf.GradientTape() as tape:
#         y_pred = model(X)
#         print(model.variables)
#         loss = tf.reduce_mean(tf.square(y_pred - y))
#     grads = tape.gradient(loss , model.variables)
#     optimizer.apply_gradients(grads_and_vars = zip(grads , model.variables))
# print(model.variables)
# 1st the linear regression
# X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
# y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

# x = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
# y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

# x = tf.constant(x)
# y = tf.constant(y)

# a = tf.Variable(initial_value = 0.)
# b = tf.Variable(initial_value = 0 , dtype = tf.float32)
# variables = [a,b]

# num_epoch = 100000
# optimizer = tf.keras.optimizers.SGD(learning_rate = 5e-4)
# for e in range(num_epoch):
#     with tf.GradientTape() as tape:
#         y_pred = a * x + b
#         loss = tf.reduce_sum(tf.square(y - y_pred))
#     grads = tape.gradient(loss , variables)
#     optimizer.apply_gradients(grads_and_vars = zip(grads , variables))
# print(a,b)
```
