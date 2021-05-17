教程
=========

如果您不熟悉DeepChem，则可能想了解一些基本的问题。如什么是DeepChem？
您为什么要关心使用它？ 一句话来概括下，DeepChem是一个科学的机器学习库。
("Chem"表示DeepChem最初专注于化学应用的历史事实，
但我们现在让deepchem更广泛地支持所有类型的科学应用)

为什么要使用DeepChem而不是其他的机器学习库？
简单来说，DeepChem中包含了大量处理科学计算的工具，如加载科学数据集、
处理数据、变换数据、切割数据、训练数据等。
DeepChem的底层使用了大量的其他机器学习框架，
如 `scikit-learn <https://scikit-learn.org/stable/>`_,  `TensorFlow <https://www.tensorflow.org/>`_, 和 `XGBoost <https://xgboost.readthedocs.io/en/latest/>`_.


我们也尝试使用 `PyTorch <https://pytorch.org/>`_和`JAX <https://github.com/google/jax>`_中进行训练模型。我们的目的使用各种可得的工具推进科学研究。

在本教程的其余部分中，我们将快速概述DeepChem的API。

DeepChem是一个大型的python模块库，因此我们不会涵盖所有内容，
但我们会给您提供足够的入门资源。 


.. contents:: 目录
    :local:

数据处理
-------------

:code:`dc.data` 模块包含了处理   :code:`Dataset` 对象的工具。
:code:`Dataset` 对象是DeepChem的核心。

A :code:`Dataset` 包含了常用的机器学习数据集。  is an abstraction of a dataset in machine learning. 
数据集中包含特征，标签，权重以及相关的标识符。
与其进一步详细说明，不如直接向您展示相应的数据集。

.. doctest:: 

   >>> import deepchem as dc
   >>> import numpy as np
   >>> N_samples = 50
   >>> n_features = 10
   >>> X = np.random.rand(N_samples, n_features)
   >>> y = np.random.rand(N_samples)
   >>> dataset = dc.data.NumpyDataset(X, y)
   >>> dataset.X.shape
   (50, 10)
   >>> dataset.y.shape
   (50,)

这里我使用了 :code:`NumpyDataset` 类，将相应的数据集会导入到内存中。
这对于对于在小规模的数据集进行探索非常方便，但是对于较大的数据集则不太方便。 

对于较大的数据集，我们推荐使用  :code:`DiskDataset` 这个类。

.. doctest::

   >>> dataset = dc.data.DiskDataset.from_numpy(X, y)
   >>> dataset.X.shape
   (50, 10)
   >>> dataset.y.shape
   (50,)


在这里示例中，我们并没有显式地指定数据目录， :code:`DiskDataset` 类会默认把数据写到临时目录中。请注意，dataset.X和dataset.y从磁盘加载数据！因此，对于较大的数据集，这可能会非常耗时。 


特征工程
-------------------

“特征化”的代码的功能是将原始输入数据转换为适合机器学习的形式。 


:code:`dc.feat` 模块包含大量对分子，分子配合物和无机晶体进行特征化的工具代码。
 
这里我们展示一个使用 :code:`dc.feat` 模块对分子进行特征化的案例。


.. doctest::

   >>> smiles = [
   ...   'O=Cc1ccc(O)c(OC)c1',
   ...   'CN1CCC[C@H]1c2cccnc2',
   ...   'C1CCCCC1',
   ...   'c1ccccc1',
   ...   'CC(=O)O',
   ... ]
   >>> properties = [0.4, -1.5, 3.2, -0.2, 1.7]
   >>> featurizer = dc.feat.CircularFingerprint(size=1024)
   >>> ecfp = featurizer.featurize(smiles)
   >>> ecfp.shape
   (5, 1024)
   >>> dataset = dc.data.NumpyDataset(X=ecfp, y=np.array(properties))
   >>> len(dataset)
   5

这里我们使用 :code:`CircularFingerprint` 代码将 SMILES 转换成了 ECFP。

ECFP由化学结构信息制成的位向量指纹，我们可以将其用作各种模型的输入。 

假设，你有一个csv文件，里面包含了SMILES和HOMO-LUMO的gap性质。

在这种情况下，通过使用 :code:`DataLoader` 代码，您可以很方便地加载数据并对其特征化。 

.. doctest::

   >>> import pandas as pd
   >>> # make a dataframe object for creating a CSV file
   >>> df = pd.DataFrame(list(zip(smiles, properties)), columns=["SMILES", "property"])
   >>> import tempfile
   >>> with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
   ...   # dump the CSV file
   ...   df.to_csv(tmpfile.name)
   ...   # initizalize the featurizer
   ...   featurizer = dc.feat.CircularFingerprint(size=1024)
   ...   # initizalize the dataloader
   ...   loader = dc.data.CSVLoader(["property"], feature_field="SMILES", featurizer=featurizer)
   ...   # load and featurize the data from the CSV file
   ...   dataset = loader.create_dataset(tmpfile.name)
   ...   len(dataset)
   5


数据分割
--------------

:code:`dc.splits` 模块包含了大量的数据分割工具。
通常，我们需要将数据集划分为训练集、验证集和测试集进行训练模型和测试模型。

下面我们会为你展示一个数据分割的使用案例。

.. doctest::

   >>> splitter = dc.splits.RandomSplitter()
   >>> # split 5 datapoints in the ratio of train:valid:test = 3:1:1
   >>> train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
   ...   dataset=dataset, frac_train=0.6, frac_valid=0.2, frac_test=0.2
   ... )
   >>> len(train_dataset)
   3
   >>> len(valid_dataset)
   1
   >>> len(test_dataset)
   1

这里我们使用 :code:`RandomSplitter`代码将数据集以3:1:1的形式划分为训练集、验证机和测试集。

注意，随机划分在小数据集或者非平衡数据集等情况，有时会高估模型的性能。
**请谨慎充分的评估模型**。 

:code:`dc.splits` 模块提供了更多的方法和算法来合理评价模型的性能， 比如交叉验证、基于分子骨架分割数据集。




模型训练和评价
-----------------------------

:code:`dc.models` 模块包含大量用于科学应用的模型。 
大部分模型可以从   :code:`dc.models.Model` 中继承而来，
我们可以通过调用 :code:`fit` 的方法对模型进行训练。

下面，我们将向你展示模型的使用方法。

.. doctest::

   >>> from sklearn.ensemble import RandomForestRegressor
   >>> rf = RandomForestRegressor()
   >>> model = dc.models.SklearnModel(model=rf)
   >>> # model training
   >>> model.fit(train_dataset)
   >>> valid_preds = model.predict(valid_dataset)
   >>> valid_preds.shape
   (1,)
   >>> test_preds = model.predict(test_dataset)
   >>> test_preds.shape
   (1,)


这里，我们使用了:code:`SklearnModel` 里面的模型，对模型进行了训练，在验证集和测试集上进行了测试。
即使您想要训练由TensorFlow或PyTorch实现的深度学习模型，也只需调用 :code:`fit` 方法即可！ 

如果您使用  :code:`dc.metrics.Metric`，则可以仅通过调用  :code:`evaluate`  方法来评估模型。 


.. doctest::

   >>> # initialze the metric
   >>> metric = dc.metrics.Metric(dc.metrics.mae_score)
   >>> # evaluate the model
   >>> train_score = model.evaluate(train_dataset, [metric])
   >>> valid_score = model.evaluate(valid_dataset, [metric])
   >>> test_score = model.evaluate(test_dataset, [metric])


更多教程
--------------

DeepChem在github上面维护着大量的`教程 <https://github.com/deepchem/deepchem/tree/master/examples/tutorials>`_, 可在Google Colab <https://colab.research.google.com/>`_上运行，该在线平台可让你很方便地执行Jupyter notebook。 

完成本入门教程后，我们建议您阅读更多的`相关教程 <https://github.com/deepchem/deepchem/tree/master/examples/tutorials>` 。 








