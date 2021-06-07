DeepChem教程4：分子指纹
==========================================================


欢迎来到DeepChem的介绍教程4——分子指纹。
我们将通过一系列的教程让你更全面的了解DeepChem工具，
从而帮助你更好的将深度学习技术应用到生命科学领域。
如果你是第一次接触这个工具，
建议先看 `入门教程1 <https://deepchembook.readthedocs.io/zh_CN/latest/examples/tutorials/01_start.html>`_。


分子可以用多种方式表示。 本教程介绍了一种称为“分子指纹”的表示方式。 这是一个非常简单的表示，非常适用于类药小分子。 


分子指纹
---------------------------

.. contents:: 目录
    :local:



分子指纹是什么？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


深度学习模型几乎总是以数字数组作为输入。 如果我们想用它们处理分子，我们需要以某种方式将每个分子表示为一个或多个数字数组。

许多（但不是全部）类型的深度学习模型要求它们的输入具有固定大小。 
这对分子来说可能是一个挑战，因为不同的分子具有不同数量的原子。 
如果我们想使用这些类型的模型，我们需要以某种方式用固定大小的数组来表示可变大小的分子。 

分子指纹当初就是为了解决这些问题而设计的。
指纹是一个固定长度的数组，其中不同的元素表示分子中存在不同的特征。
如果两个分子具有相似的指纹，则表明它们包含许多相同的特征，因此很可能具有相似的化学性质。

DeepChem 支持一种特定类型的指纹，称为“扩展连接指纹（Extended Connectivity Fingerprint）”，简称“ECFP”。
它们有时也被称为“圆形指纹(circular fingerprints)”。 
ECFP 算法首先根据原子的直接性质和键对原子进行分类。 每个独特的图案都是一个特征。
例如，“与两个氢原子和两个重原子键合的碳原子”将是一个特征，对于包含该特征的任何分子，指纹的特定元素设置为 1。 
然后它通过查看更大的圆形邻域来迭代地识别新特征。 
与其他两个特定特征结合的一个特定特征成为更高级别的特征，并且为包含它的任何分子设置相应的元素。 这会持续固定数量的迭代，最常见的是两次(ECFP4)。 

让我们看一下使用 ECFP 进行特征化的数据集。 

.. code-block:: python 

    import deepchem as dc 
    tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='ECFP')
    train_dataset, valid_dataset, test_dataset = datasets
    print(train_dataset)


输出：

.. code-block:: console 

    <DiskDataset X.shape: (6264, 1024), y.shape: (6264, 12), w.shape: (6264, 12), task_names: ['NR-AR' 'NR-AR-LBD' 'NR-AhR' ... 'SR-HSE' 'SR-MMP' 'SR-p53']>


The feature array X has shape (6264, 1024). That means there are 6264 samples in the training set. Each one is represented by a fingerprint of length 1024. Also notice that the label array y has shape (6264, 12): this is a multitask dataset. Tox21 contains information about the toxicity of molecules. 12 different assays were used to look for signs of toxicity. The dataset records the results of all 12 assays, each as a different task.


特征数组 X的大小是(6264, 1024)。 
这意味着训练集中有 6264 个样本。 
每个都由长度为 1024 的指纹表示。
还要注意标签数组 y 的形状为 (6264, 12)：这是一个多任务（12个任务）数据集。
Tox21 数据集包含分子的各种（12种）毒性的信息。
 

 我们来看看权重数组。

 .. code-block:: python 

    print(train_dataset.w)


输出：

.. code-block:: console 

    [[1.04502242 1.03632599 1.12502653 ... 1.05576503 1.17464996 1.05288369]
    [1.04502242 1.03632599 1.12502653 ... 1.05576503 1.17464996 1.05288369]
    [1.04502242 1.03632599 1.12502653 ... 1.05576503 0.         1.05288369]
    ...
    [1.04502242 0.         1.12502653 ... 1.05576503 6.7257384  1.05288369]
    [1.04502242 1.03632599 1.12502653 ... 1.05576503 6.7257384  1.05288369]
    [1.04502242 1.03632599 1.12502653 ... 0.         1.17464996 1.05288369]]



请注意，列表中的某些元素为 0。权重为0用于指示缺失数据。
并非所有的测试都在每个分子进行。
将样本或样本/任务对的权重设置为 0 会导致它在拟合和评估期间被忽略。 它不会对损失函数或其他指标产生影响。

大多数其他权重都接近于 1，但不完全是 1。这样做是为了平衡每个任务上正负样本的总体权重。 
在训练模型时，我们希望 12 个任务中的每一个贡献均等，并且在每个任务上，我们希望对正样本和负样本赋予相同的权重。
否则，模型可能只会了解到大多数训练样本是无毒的，会偏向于将其他分子识别为无毒。 



基于分子指纹训练模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


让我们训练一个模型。 在之前的教程中，我们使用过图卷积模型（ GraphConvModel），这是一个相当复杂的架构，需要一组复杂的输入。
因为指纹非常简单，只是一个固定长度的数组，我们可以使用更简单的模型类型。 

 .. code-block:: python 

    model = dc.models.MultitaskClassifier(n_tasks=12, n_features=1024, layer_sizes=[1000])


MultitaskClassifier 是一个简单的全连接层模型。 
在这个例子中，我们告诉它使用一个宽度为 1000 的隐藏层。我们还告诉它每个输入将有 1024 个特征，并且它应该为 12 个不同的任务生成预测。

为什么不为每个任务训练一个单独的模型？ 我们可以这样做，但事实证明，为多个任务训练单个模型通常效果更好。 我们将在后面的教程中看到一个例子。

接下来让我们训练和评价模型。



 .. code-block:: python 

    import numpy as np
    import deepchem as dc 
    model = dc.models.MultitaskClassifier(n_tasks=12, n_features=1024, layer_sizes=[1000])
    model.fit(train_dataset, nb_epoch=10)
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    print('training set score:', model.evaluate(train_dataset, [metric], transformers))
    print('test set score:', model.evaluate(test_dataset, [metric], transformers))

输出：

.. code-block:: console 

    training set score: {'roc_auc_score': 0.9573709576148927}
    test set score: {'roc_auc_score': 0.6812477027425125}



对于这样一个简单的模型和特征化来说，效果不错。 更复杂的模型在这个数据集上做得稍微好一些，但并没有好得多。 






