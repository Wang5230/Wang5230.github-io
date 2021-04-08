---
title: TensorFlow1.x学习笔记
date: 2020-08-26 10:43:57
tags: python
---

## debug

### module 'tensorflow' has no attribute 'Session'(已修复)

在新版本中Tensorflow 2.0版本中已经移除了Session这一模块，改换运行代码以及对于graph的语法皆以变化，语法近似pytorch

``` python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
sess= tf.compat.v1.Session(config=config)
```

## 建立一个简单的回归分析模型

1. 首先使用placeholder为输入与输出创建占位符，同时为权重和截距创建合适的变量

``` pyhton
x = tf.placeholder(tf.float32 , shape = [None ,3]) #输入
y_ture = tf.placeholder(tf.float32 , shape = None) #实际值
w = tf.Variable([[0,0,0]] , dtype = tf.float32 , 
name = 'weights')  #权重
b = tf.Variable(0 , dtype = tf.float32 , name = 'bias') #截距
y_pred = tf.matmul(w , tf.transpose(x)) + b #预测值
```

2. 接下来，需要来评估模型的性能，为了刻画预测值与真实值的差异，需要订一个反应“距离”的度量，一般称为**损失函数**，通过寻找一组参数来最小化损失函数优化模型。一般使用**MSE(均方差)和交叉熵**

``` pyhton
#-----------------MRE
loss = tf.reduce_mean(tf.square(y_ture - y_pred)) #MSE
#----------------交叉熵
loss = tf.nn.sigmod_cross_entropy_with_logits(lables = y_ture , logits = y_pred)
loss = tf.reduce_mean(loss)
```

3. 接下来需要明白如何最小化损失函数，一般使用**梯度下降法**，尽管可能陷入局部最优，但是一般都是足够好的。

4. 由于计算整个样本集合可能会很慢，所以需要使用一些采样方法，对样本的子集进行采样，一般规模在50~500个一次，过小的样本会使得硬件利用率降低，而且会使得**目标函数**产生较大的波动，但有时波动也是有益的，因为能够使得参数跳跃到新的局部最优值，TF中通过向图中添加新的操作然后使用自动差分来计算梯度，需要设置的就是学习率，来确定每次更新迭代的量，一般更倾向于设置较小的学习率，以防止跳过了局部最优解，但过低会导致损失函数减少的十分慢。

``` python
optimizer = tf.train.GradientDsecentOptimizer(learning_rate)
train = optimizer.minimize(loss)
```

### 线性回归

使用numpy生成数据，创建了三个特征向量样本，每个样本内积乘以一组权重加上偏差项再加上噪声得出结果，通过优化模型找到最佳参数

``` python

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
x_data = np.random.randn(2000 , 3)
w_real = [0.3 , 0.5 , 0.1]
b_real = -0.2

noise = np.random.randn(1,2000)*0.1
y_data = np.matmul(w_real , x_data.T) + b_real + noise
#生成数据
NUM_STEPS = 11
wb = []
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32 , shape = [None , 3])
    y_true = tf.placeholder(tf.float32 ,shape=None)
    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0,0,0]] , dtype = tf.float32 , name = 'weights')
        b = tf.Variable(0 , dtype = tf.float32 , name = 'bias')
        y_pred = tf.matmul(w , tf.transpose(x)) + b

    with tf.name_scope('loss') as scope:
        loss = tf.reduce_mean(tf.square(y_true - y_pred))

    with tf.name_scope('train') as scope:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEPS):
            sess.run(train,feed_dict={x :x_data , y_true : y_data})
            if step % 5 == 0:
                print(step , sess.run([w,b]))
                wb.append(sess.run([w,b]))
```

### 逻辑回归

一般用来进行二分类任务，输出一个0-1之间地离散二值结果，通过sigmoid函数来将数值映射到0到1之间，之后通过阈值分类器来将0到1之间的值转换为0或1，与之前代码的唯一不同是损失函数部分

``` pyhton
with tf.name_scope('loss') as scope:
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y_true , logits = y_pred)
loss = tf.reduce_mean(loss)
```

## CNN

层中的神经元只与前一层中的一小块区域连接，而不是采取全连接方式，主要由输入层，卷积层，ReLU层，池化层和全连接层构成(一般将卷积层和ReLU层一起称为卷积层)具体说来，卷积层和全连接层（CONV/FC）对输入执行变换操作的时候，不仅会用到激活函数，还会用到很多参数，即神经元的权值w和偏差b；而ReLU层和池化层则是进行一个固定不变的函数操作。卷积层和全连接层中的参数会随着梯度下降被训练，这样卷积神经网络计算出的分类评分就能和训练集中的每个图像的标签吻合了。

### 卷积层

参数由一些可学习的滤波器集合构成，只观察输入数据中的一小部分，由于卷积有“权值共享”这样的特性，可以降低参数的数量，防止参数过多造成过拟合。每个神经元连接的空间大小叫做神经元的感受野，深度与输入相同，但宽高是局部的。输出数据体和使用的滤波器的数量一致，将沿着深度方向排列。有时候在输入数据体的边缘使用0进行填充，使得滤波器可以平滑地在数据上滑动，一般是用来保持数据体的空间尺寸使输入输出宽高相同。如果在一个深度切片中的所有权重都使用同一个权重向量，那么卷积层的前向传播在每个深度切片中可以看做是在计算神经元权重和输入数据体的卷积（这就是“卷积层”名字由来）。这也是为什么总是将这些权重集合称为滤波器（filter）（或卷积核（kernel）），因为它们和输入进行了卷积。简单来说，就是一个在原始数据上以步长为长度不断移动的一个矩阵，层数与原始数据一致，将每个对应位置的数据进行乘积并且将每层的值相加，即为输出的一个数据，例如RGB通道下就是三层滤波器所得相加填入新的矩阵当中。

### 池化层

用来简化数据，减少计算量，有最大池化和平均池化，使用较多的是最大池化，通过找出某一个区域内的最大值/平均值来缩小数据，一般使用的参数是f=2，p=2 恰好为缩小数据为原来的一半，并且反向传播没有参数适用于池化，这一过程是静态过程，参数都是手动设定，或是交叉验证得到的，只用于简化数据，提取特征。

### 全连接层

将池化层简化后的数据进行提取，将特征整合到一起，输出为一个值，主要作用是减小特征位置对于结果产生的影响，将结果进行分类，一般不止一层，需要将提取出来的特征神经元激活后，通过不同神经元激活的组合，再提取出结果

简单的卷积实现mnist识别,将测试过程分为每个大小为1000幅图的十块

``` python
import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
DATA_DIR = '/tmp/data'
NUM_STEPS = 10000
MINIBATCH_SIZE = 100
tf.disable_eager_execution()
def weight_variable(shape):
    initial = tf.truncated_normal(shape , stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1 , shape = shape)
    return tf.Variable(initial)
def conv2d (x , w):
    return tf.nn.conv2d(x , w , strides = [1, 1 ,1 ,1] , padding = 'SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x , ksize = [1 , 2, 2,1], strides = [1,2,2,1] , padding='SAME')
def conv_layer(input , shape):
    w = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input,w)+b)
def full_layer(input , size):
    in_size = int(input.get_shape()[1])
    w = weight_variable([in_size,size])
    b = bias_variable([size])
    return tf.matmul(input,w) + b
x = tf.placeholder(tf.float32 , shape=[None , 784])
y_ = tf.placeholder(tf.float32 , shape=[None , 10])
x_image = tf.reshape(x,[-1,28,28,1])
conv1 = conv_layer(x_image , shape = [5,5,1,32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool , shape = [5,5,32,64])
conv2_pool = max_pool_2x2(conv2)

conv2_flat = tf.reshape(conv2_pool, [-1,7*7*64])
full_1 = tf.nn.relu(full_layer(conv2_flat , 1024))

keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1 , keep_prob = keep_prob)

y_conv = full_layer(full1_drop , 10)

minst = input_data.read_data_sets(DATA_DIR , one_hot=True)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_conv , labels = y_) )
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv , 1) , tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(NUM_STEPS):
        batch = minst.train.next_batch(50)

        if i % 100 == 0:
            train_accuracy = sess.run(accuracy , feed_dict={x:batch[0] , y_:batch[1] , keep_prob : 1.0})
            print("step {} , train accuracy {}".format(i,train_accuracy))
        sess.run(train_step , feed_dict={x:batch[0] , y_:batch[1] , keep_prob : 0.5})
        X = minst.test.images.reshape(10,1000,784)
        Y = minst.test.labels.reshape(10,1000,10)
        test_accuracy = np.mean([sess.run(accuracy , feed_dict={x:X[i] , y_:Y[i] , keep_prob : 1.0}) for i in range(10)])
    print("test accuracy: {}".format(test_accuracy))
```

### ResNet

深度残差网络（Deep residual network, ResNet）的提出是CNN图像史上的一件里程碑事件.通过引入残差学习的方法来解决网络在随着深度增加时出现的准确度饱和下降，梯度爆炸以及消失的难以训练的问题。对于一个堆积结构，在输入x时，学到的特征为H(x)，现在希望其学到残差即 H(x)-x,这样即使残差为0时，也仅是在层之间做了恒等映射，至少不会性能下降，残差使得其会在本有的特征上学习到新的特征，有点类似电路中的短路`