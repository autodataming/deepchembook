DeepChem教程5：使用PyTorch或者TensorFlow 创建模型 
==========================================================


欢迎来到DeepChem的介绍教程5——使用PyTorch或者TensorFlow 创建模型。
我们将通过一系列的教程让你更全面的了解DeepChem工具，
从而帮助你更好的将深度学习技术应用到生命科学领域。
如果你是第一次接触这个工具，
建议先看 `入门教程1 <https://deepchembook.readthedocs.io/zh_CN/latest/examples/tutorials/01_start.html>`_。



在到目前为止的教程中，我们使用了 DeepChem 提供的标准模型。 
这对许多应用程序来说都很好，但迟早你会想要用你自己定义的架构来创建一个全新的模型。
DeepChem 提供了与 TensorFlow (Keras) 和 PyTorch 的集成使用的接口，因此你可以在deepchem中使用PyTorch或者TensorFlow 创建模型 

使用PyTorch或者TensorFlow 创建模型 
-----------------------------------------

.. contents:: 目录
    :local:



KerasModel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


KerasModel is a subclass of DeepChem's Model class. 
It acts as a wrapper around a tensorflow.keras.Model. 
Let's see an example of using it. For this example, we create a simple sequential model consisting of two dense layers.


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


TorchModel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TorchModel 的工作方式与 KerasModel 类似，不同之处在于它封装的是 torch.nn.Module。
让我们使用 PyTorch 创建另一个模型，就像之前的模型一样，并在相同的数据上对其进行训练。 

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











