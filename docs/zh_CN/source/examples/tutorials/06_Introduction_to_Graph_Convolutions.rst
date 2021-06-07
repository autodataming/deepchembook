DeepChem教程06：图卷积（Graph Convolutions ）网络的介绍
==========================================================


欢迎来到DeepChem的介绍教程06——图卷积（Graph Convolutions ）网络的介绍。
我们将通过一系列的教程让你更全面的了解DeepChem工具，
从而帮助你更好的将深度学习技术应用到生命科学领域。
如果你是第一次接触这个工具，
建议先看 `入门教程1 <https://deepchembook.readthedocs.io/zh_CN/latest/examples/tutorials/01_start.html>`_。



在本教程中，我们将了解有关“图卷积（graph convolutions）”的更多信息。
这是处理分子数据的最强大的深度学习工具之一。 这样做的原因是分子可以自然地被视为Graph。 

请注意，我们从高中开始习惯的将分子和标准化学图关联在一起。
在本教程的其余部分，我们将更详细地深入研究这种关系。 这将使我们更深入地了解这些系统的工作原理。 




图卷积（Graph Convolutions ）网络
-----------------------------------------

.. contents:: 目录
    :local:

什么是图卷积网络？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
考虑一种常用于处理图像的标准卷积神经网络 (CNN)。 输入是像素网格。 
每个像素都有一个数据值向量，例如红色、绿色和蓝色通道。 
数据通过一系列卷积层。 每一层都将来自一个像素及其相邻像素的数据组合起来，为该像素生成一个新的数据向量。
早期层检测小尺度局部模式，而后期层检测更大、更抽象的模式。 
通常，卷积层与池化层交替进行，池化层在局部区域执行某些操作，例如最大值或最小值。


图卷积和其类似，但它们在图上操作。 它们以图中每个节点的数据向量开始（例如，该节点代表的原子的化学性质）。 
卷积层和池化层结合来自连接节点（例如，相互结合的原子）的信息，为每个节点生成一个新的数据向量。 





训练图卷积网络
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's use the MoleculeNet suite to load the Tox21 dataset. To featurize the data in a way that graph convolutional networks can use, we set the featurizer option to 'GraphConv'. The MoleculeNet call returns a training set, a validation set, and a test set for us to use. It also returns tasks, a list of the task names, and transformers, a list of data transformations that were applied to preprocess the dataset. (Most deep networks are quite finicky and require a set of data transformations to ensure that training proceeds stably.)



KerasModel 是 DeepChem 的 Model 类的子类。
它可以封装tensorflow.keras.Model。

让我们看一个使用它的例子。 
对于此示例，我们创建了一个由两个连接层组成的简单顺序模型。 


.. code-block:: python 

    import deepchem as dc
    import tensorflow as tf

    keras_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(1)
    ])
    model = dc.models.KerasModel(keras_model, dc.models.losses.L2Loss())


在这个例子中，我们使用了 Keras Sequential 类进行构建模型。 
我们的模型由带有ReLU激活函数的全链接层组成，提供正则化的 50% dropout 和产生标量输出的最后一层组成。
我们还需要指定在训练模型时使用的损失函数，在本例中为 L2LOSS 函数。 
我们现在可以像使用任何其他 DeepChem 模型一样训练和评估模型。 例如，让我们加载 Delaney 溶解度数据集。
 这个模型基于extended-connectivity fingerprints (ECFPs)预测分子的溶解度的表现如何呢？

.. code-block:: python 

    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='ECFP', splitter='random')
    train_dataset, valid_dataset, test_dataset = datasets
    model.fit(train_dataset, nb_epoch=50)
    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
    print('training set score:', model.evaluate(train_dataset, [metric]))
    print('test set score:', model.evaluate(test_dataset, [metric]))


输出：
 

.. code-block:: console 

    training set score: {'pearson_r2_score': 0.9795690217950392}
    test set score: {'pearson_r2_score': 0.725184181624837}

通过对模型的架构进行调整，模型的预测能力增强。


.. code-block:: python 

    import torch
    import deepchem as dc 

    pytorch_model = torch.nn.Sequential(
        torch.nn.Linear(1024, 1000),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1000, 1)
    )
    model = dc.models.TorchModel(pytorch_model, dc.models.losses.L2Loss())

    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='ECFP', splitter='random')
    train_dataset, valid_dataset, test_dataset = datasets

    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

    model.fit(train_dataset, nb_epoch=50)
    print('training set score:', model.evaluate(train_dataset, [metric]))
    print('test set score:', model.evaluate(test_dataset, [metric]))


输出：
 
.. code-block:: console 

    training set score: {'pearson_r2_score': 0.9797902109595925}
    test set score: {'pearson_r2_score': 0.7014179421837455}











