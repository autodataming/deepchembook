案例
========

这里我们会展示DeepChem的一些使用案例。

- We match against doctest's :code:`...` wildcard on code where output is usually ignored
- We often use threshold assertions (e.g: :code:`score['mean-pearson_r2_score'] > 0.92`),
  as this is what matters for model training code.

.. contents:: 目录
    :local:


在进入案例之前，我们需要导入一些常用的模块。

.. doctest:: *

    >>> import numpy as np
    >>> import tensorflow as tf
    >>> import deepchem as dc
    >>>
    >>> # Run before every test for reproducibility
    >>> def seed_all():
    ...     np.random.seed(123)
    ...     tf.random.set_seed(123)

.. testsetup:: *

    import numpy as np
    import tensorflow as tf
    import deepchem as dc

    # Run before every test for reproducibility
    def seed_all():
        np.random.seed(123)
        tf.random.set_seed(123)


Delaney (ESOL)
----------------

Delaney（ESOL）是一个回归数据集，其中包含1128种化合物的结构和水溶性数据，
收录在`MoleculeNet <./moleculenet.html>`_的数据集合中。
该数据集被广泛用于建立基于分子结构（以SMILES字符串编码）估算溶解度的机器学习模型。
 
我们会使用数据集中的 :code:`smiles` 字段进行训练模型预测实验测得的溶剂化能(:code:`expt`)。

多任务回归模型
^^^^^^^^^^^^^^^^^^


首先，我们会使用 :func:`load_delaney() <deepchem.molnet.load_delaney>` 函数进行加载数据；
然后通过 :class:`MultitaskRegressor <deepchem.models.MultitaskRegressor>` 类的fit的方法进行训练模型。


.. doctest:: delaney

    >>> seed_all()
    >>> # Load dataset with default 'scaffold' splitting
    >>> tasks, datasets, transformers = dc.molnet.load_delaney()
    >>> tasks
    ['measured log solubility in mols per litre']
    >>> train_dataset, valid_dataset, test_dataset = datasets
    >>>
    >>> # We want to know the pearson R squared score, averaged across tasks
    >>> avg_pearson_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
    >>>
    >>> # We'll train a multitask regressor (fully connected network)
    >>> model = dc.models.MultitaskRegressor(
    ...     len(tasks),
    ...     n_features=1024,
    ...     layer_sizes=[500])
    >>>
    >>> model.fit(train_dataset)
    0...
    >>>
    >>> # We now evaluate our fitted model on our training and validation sets
    >>> train_scores = model.evaluate(train_dataset, [avg_pearson_r2], transformers)
    >>> assert train_scores['mean-pearson_r2_score'] > 0.7, train_scores
    >>>
    >>> valid_scores = model.evaluate(valid_dataset, [avg_pearson_r2], transformers)
    >>> assert valid_scores['mean-pearson_r2_score'] > 0.3, valid_scores


图卷积模型
^^^^^^^^^^^^^^


对于Delaney数据集默认的特征化`featurizer <./featurizers.html>`_ 方式是 :code:`ECFP`（Extended-connectivity fingerprints） 。
对于图卷积模型 :class:`GraphConvModel <deepchem.models.GraphConvModel>`，
我们在加载数据的时候需要显示指定特征化的方式为:code:`featurizer='GraphConv'`。


.. doctest:: delaney

    >>> seed_all()
    >>> tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv')
    >>> train_dataset, valid_dataset, test_dataset = datasets
    >>>
    >>> model = dc.models.GraphConvModel(len(tasks), mode='regression', dropout=0.5)
    >>>
    >>> model.fit(train_dataset, nb_epoch=30)
    0...
    >>>
    >>> # We now evaluate our fitted model on our training and validation sets
    >>> train_scores = model.evaluate(train_dataset, [avg_pearson_r2], transformers)
    >>> assert train_scores['mean-pearson_r2_score'] > 0.5, train_scores
    >>>
    >>> valid_scores = model.evaluate(valid_dataset, [avg_pearson_r2], transformers)
    >>> assert valid_scores['mean-pearson_r2_score'] > 0.3, valid_scores


ChEMBL数据集
--------------


`ChEMBL <https://www.ebi.ac.uk/chembl/>`_ 数据集是手动收集整理具有类药性质的生物活性分子的数据库。
它包含了化学、活性、基因组数据（靶点数据），目的是加速从基因组信息寻找有效的药物分子。
该数据集的22.1版本也已经整合到`MoleculeNet <./moleculenet.html>`_的数据集合中, 里面包含了2个类别 “sparse” 和 “5thresh”  。
“sparse”是一个大的数据集，包含了 244,245 化合物的化合物的信息。
正如名字所示的那样，这个数据集中数据非常稀疏，大部分化合物仅仅有一个靶标的活性数据。 

 “5thresh” 是一个更小的数据集，包含了23,871 化合物的信息，每个化合物至少有5个靶标的活性数据。

Examples of training models on `ChEMBL`_  dataset included in MoleculeNet.

下面是基于`ChEMBL <https://www.ebi.ac.uk/chembl/>`_ 数据集进行训练的案例。


多任务回归模型
^^^^^^^^^^^^^^^^^^

.. doctest:: chembl

    >>> seed_all()
    >>> # Load ChEMBL 5thresh dataset with random splitting
    >>> chembl_tasks, datasets, transformers = dc.molnet.load_chembl(
    ...     shard_size=2000, featurizer="ECFP", set="5thresh", split="random")
    >>> train_dataset, valid_dataset, test_dataset = datasets
    >>> len(chembl_tasks)
    691
    >>> f'Compound train/valid/test split: {len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)}'
    'Compound train/valid/test split: 19096/2387/2388'
    >>>
    >>> # We want to know the RMS, averaged across tasks
    >>> avg_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
    >>>
    >>> # Create our model
    >>> n_layers = 3
    >>> model = dc.models.MultitaskRegressor(
    ...     len(chembl_tasks),
    ...     n_features=1024,
    ...     layer_sizes=[1000] * n_layers,
    ...     dropouts=[.25] * n_layers,
    ...     weight_init_stddevs=[.02] * n_layers,
    ...     bias_init_consts=[1.] * n_layers,
    ...     learning_rate=.0003,
    ...     weight_decay_penalty=.0001,
    ...     batch_size=100)
    >>>
    >>> model.fit(train_dataset, nb_epoch=5)
    0...
    >>>
    >>> # We now evaluate our fitted model on our training and validation sets
    >>> train_scores = model.evaluate(train_dataset, [avg_rms], transformers)
    >>> assert train_scores['mean-rms_score'] < 10.00
    >>>
    >>> valid_scores = model.evaluate(valid_dataset, [avg_rms], transformers)
    >>> assert valid_scores['mean-rms_score'] < 10.00



图卷积模型 
^^^^^^^^^^^^^^

.. doctest:: chembl

    >>> # Load ChEMBL dataset
    >>> chembl_tasks, datasets, transformers = dc.molnet.load_chembl(
    ...    shard_size=2000, featurizer="GraphConv", set="5thresh", split="random")
    >>> train_dataset, valid_dataset, test_dataset = datasets
    >>>
    >>> # RMS, averaged across tasks
    >>> avg_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
    >>>
    >>> model = dc.models.GraphConvModel(
    ...    len(chembl_tasks), batch_size=128, mode='regression')
    >>>
    >>> # Fit trained model
    >>> model.fit(train_dataset, nb_epoch=5)
    0...
    >>>
    >>> # We now evaluate our fitted model on our training and validation sets
    >>> train_scores = model.evaluate(train_dataset, [avg_rms], transformers)
    >>> assert train_scores['mean-rms_score'] < 10.00
    >>>
    >>> valid_scores = model.evaluate(valid_dataset, [avg_rms], transformers)
    >>> assert valid_scores['mean-rms_score'] < 10.00



    