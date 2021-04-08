---
title: 'batch,epoch,iteration'
date: 2020-10-29 18:33:13
tags:
    - python
    - machinelearning
---
1. batch size：简单来说，batch size就是每次训练向模型中传入的数据量的多少，过小的话会导致梯度不明显，模型下降速度慢，过大的话尽管速度会更快但会导致需要更多的epoch来获得更好的结果，与batch本身的节约内存空间的目的相悖，因此需要选择一个合适的batch大小来进行训练。

2. iteration:意思为迭代，一个iteration等于将batch size中的数据迭代一遍

3. epoch:意思为周期，一个周期相当于将样本中的所有数据遍历一遍。例如样本数量为1000，batch size为10，那么就需要100此iteration来完成一个epoch。
