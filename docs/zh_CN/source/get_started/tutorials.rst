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

The :code:`dc.data` module contains utilities to handle :code:`Dataset`
objects. These :code:`Dataset` objects are the heart of DeepChem.
A :code:`Dataset` is an abstraction of a dataset in machine learning. That is,
a collection of features, labels, weights, alongside associated identifiers.
Rather than explaining further, we'll just show you.

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

Here we've used the :code:`NumpyDataset` class which stores datasets in memory.
This works fine for smaller datasets and is very convenient for experimentation,
but is less convenient for larger datasets. For that we have the :code:`DiskDataset` class.

.. doctest::

   >>> dataset = dc.data.DiskDataset.from_numpy(X, y)
   >>> dataset.X.shape
   (50, 10)
   >>> dataset.y.shape
   (50,)

In this example we haven't specified a data directory, so this :code:`DiskDataset` is written
to a temporary folder. Note that :code:`dataset.X` and :code:`dataset.y` load data
from disk underneath the hood! So this can get very expensive for larger datasets.


特征工程
-------------------

"Featurizer" is a chunk of code which transforms raw input data into a processed
form suitable for machine learning. The :code:`dc.feat` module contains an extensive collection
of featurizers for molecules, molecular complexes and inorganic crystals.
We'll show you the example about the usage of featurizers.

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

Here, we've used the :code:`CircularFingerprint` and converted SMILES to ECFP.
The ECFP is a fingerprint which is a bit vector made by chemical structure information
and we can use it as the input for various models.

And then, you may have a CSV file which contains SMILES and property like HOMO-LUMO gap. 
In such a case, by using :code:`DataLoader`, you can load and featurize your data at once.

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

The :code:`dc.splits` module contains a collection of scientifically aware splitters.
Generally, we need to split the original data to training, validation and test data
in order to tune the model and evaluate the model's performance.
We'll show you the example about the usage of splitters.

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

Here, we've used the :code:`RandomSplitter` and splitted the data randomly
in the ratio of train:valid:test = 3:1:1. But, the random splitting sometimes
overestimates  model's performance, especially for small data or imbalance data.
Please be careful for model evaluation. The :code:`dc.splits` provides more methods
and algorithms to evaluate the model's performance appropriately, like cross validation or
splitting using molecular scaffolds.


模型训练和评价
-----------------------------

The :code:`dc.models` conteins an extensive collection of models for scientific applications. 
Most of all models inherits  :code:`dc.models.Model` and we can train them by just calling :code:`fit` method.
You don't need to care about how to use specific framework APIs.
We'll show you the example about the usage of models.

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

Here, we've used the :code:`SklearnModel` and trained the model.
Even if you want to train a deep learning model which is implemented
by TensorFlow or PyTorch, calling :code:`fit` method is all you need!

And then, if you use :code:`dc.metrics.Metric`, you can evaluate your model
by just calling :code:`evaluate` method.

.. doctest::

   >>> # initialze the metric
   >>> metric = dc.metrics.Metric(dc.metrics.mae_score)
   >>> # evaluate the model
   >>> train_score = model.evaluate(train_dataset, [metric])
   >>> valid_score = model.evaluate(valid_dataset, [metric])
   >>> test_score = model.evaluate(test_dataset, [metric])


更多教程
--------------

DeepChem maintains `an extensive collection of addition tutorials <https://github.com/deepchem/deepchem/tree/master/examples/tutorials>`_ that are meant to
be run on `Google Colab <https://colab.research.google.com/>`_, an online platform that allows you to execute Jupyter notebooks.
Once you've finished this introductory tutorial, we recommend working through these more involved tutorials.






