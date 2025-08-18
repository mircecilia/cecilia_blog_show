---
date : '2025-08-18T00:58:25+08:00'
draft : false
title : 'Attention is all you need'
categories:
  - transformer
---

## Embedding matrix

  对于 llm 来说，人类的所有语言必须被转换为数字才能被接受，而这一转化过程所需的数据是由不同的训练任务训练后得到的，transformer 所要求的张量不仅需要表达这个词的语意，同时要求包含这个词的位置信息

  而这两个需求在最初设想时是被分开的，即：先通过特定任务训练 token 对应的张量表达相关语意，再训练张量用于描述位置信息，二者相加从而得到 transformer 最初所需的张量，但这种做法会破坏远程衰减性，损伤原句最初表达的意思，因此，现多使用 RoPE (Rotary Positional Embeddings) 对 Q、K 张量进行旋转，从而将位置信息融入查询矩阵，键矩阵中

## Query , Key and value
  
  在获得 token 对应的张量后，transformer 将使用训练得到的权重矩阵 $Wq , Wk , Wv$ 与其相乘，最终得到 Q , K , V 三组张量， 再进行 $ Softmax(\frac{Q*K^T}{\sqrt{d_k}})V $ ，同时掩蔽一些注意力，从而使后面的 token 无法影响前面 token 的输出，得到的结果即不同 tokens 之间的加权相关性，将其与每个最初始的张量相加，第一次的注意力更新就完成了

## Multilayer perceptron
  
  在 transformers 层更新后，新的张量将进入 MLP 层进行处理，通过 linear 进行线性变换，再通过 RELU 等函数进行剪裁，处理后的向量再通过线性变换恢复为原来的维度，其中所有参与线性变换的矩阵参数皆通过梯度下降及反向传播进行学习

## Backpropagation
  
  反向传播的灵感来源于链式求导法则中大量重复的计算结果，在此我们使用识别手写数字这一只包含线性神经元的项目进行举例：

  对于任意一级的神经元，其都由权重矩阵、bias、激活函数 (Softmax) 以及概率输出构成，则对于任意可调参数，都可从后向前由链式求导法则计算其导数，在固定学习率后，对每个参数都减去学习率乘以导数，即可使结果的损失函数最小化，这也就是通过数据集训练模型的方法