DeepChem教程3：MoleculeNet的介绍
==========================================================
欢迎来到DeepChem的介绍教程3——MoleculeNet数据集合的介绍。
我们将通过一系列的教程让你更全面的了解DeepChem工具，
从而帮助你更好的将深度学习技术应用到生命科学领域。
如果你是第一次接触这个工具，
建议先看 `入门教程1 <https://deepchembook.readthedocs.io/zh_CN/latest/examples/tutorials/01_start.html>`_。


DeepChem 最强大的功能之一是它带有处理好可拓展的据集以供模型使用。 
 DeepChem开发社区维护 `MoleculeNet <https://pubs.rsc.org/--/content/articlehtml/2018/sc/c7sc02664a>`_ 数据集集合，
  包含了大量各种用于机器学习的科学数据集。 
最初MoleculeNet 数据集合包含了17个数据集，主要集中在分子性质。
在过去几年中，MoleculeNet 已经发展成为更广泛的科学数据集集合促进机器学习工具的广泛使用和开发。 

这些数据集与 DeepChem 套件的其余部分集成在一起，因此你可以通过 dc.molnet 子模块中的函数方便地访问这些数据集。 
在学习本系列教程时，您已经看到了一些数据加载器的示例。 MoleculeNet 套件的完整文档可在我们的 `文档 <https://deepchem.readthedocs.io/en/latest/moleculenet.html>`_ 中找到。



MoleculeNet的介绍
---------------------------

.. contents:: 目录
    :local:



MoleculeNet Overview
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

in the last two tutorials we loaded the Delaney dataset of molecular solubilities. Let's load it one more time.
在前2个教程中，我们加载了分子溶解度的Delaney数据集。让我们再加载一次。

.. code-block:: python 

    import deepchem as dc 
    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv', splitter='random')

请注意，我们调用的 dc.molnet.load_delaney 的加载器函数位于MoleculeNet加载器的 dc.molnet 子模块中。 
让我们来看看我们目前MoleculeNet中有多少可用的加载器。 

.. code-block:: python 

    import deepchem as dc 
    loadmethods = [method for method in dir(dc.molnet) if "load_" in method ]
    print(loadmethods)
    print(len(loadmethods))

输出：

.. code-block:: console 

    ['load_Platinum_Adsorption', 'load_bace_classification', 'load_bace_regression', 'load_bandgap', 'load_bbbc001', 'load_bbbc002', 'load_bbbp', 'load_cell_counting', 'load_chembl', 'load_chembl25', 'load_clearance', 'load_clintox', 'load_delaney', 'load_factors', 'load_function', 'load_hiv', 'load_hopv', 'load_hppb', 'load_kaggle', 'load_kinase', 'load_lipo', 'load_mp_formation_energy', 'load_mp_metallicity', 'load_muv', 'load_nci', 'load_pcba', 'load_pdbbind', 'load_perovskite', 'load_ppb', 'load_qm7', 'load_qm8', 'load_qm9', 'load_sampl', 'load_sider', 'load_sweet', 'load_thermosol', 'load_tox21', 'load_toxcast', 'load_uspto', 'load_uv', 'load_zinc15']
    41

我们看到里面包含了各种数据加载器，我们还会不停的向里面增加新的数据集。在笔者使用的版本中有41个不同的数据加载器。





MoleculeNet 数据集分类
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


MoleculeNet 中有许多不同的数据集。
让我们快速浏览一下可里面的不同类型的数据集。

我们将把数据集分成不同的类别并列出属于这些类别的加载器，通过加载器可以加载相应的数据集。

有关这些数据集的更多详细信息，请访问 `MoleculeNet文档 <https://deepchem.readthedocs.io/en/latest/moleculenet.html>`_。
最初的 `MoleculeNet 论文 <https://pubs.rsc.org/--/content/articlehtml/2018/sc/c7sc02664a>`_  提供了部分数据集的信息。
 我们在下面将这些论文中介绍的数据集标记为“V1”，论文中没有记录的数据集标记为“V2”。 



量子力学数据集 
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

MoleculeNet中的量子力学数据集 （quantum mechanical datasets） 包含各种量子力学性质预测任务。
当前的一组量子力学数据集包括 QM7、QM7b、QM8、QM9。 相应的数据加载器如下：



-  dc.molnet.load_qm7: V1
-  dc.molnet.load_qm7b_from_mat: V1
-  dc.molnet.load_qm8: V1
-  dc.molnet.load_qm9: V1




物理化学数据集
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

物理化学数据集集合包含用于预测分子的各种物理特性的任务。 相应的数据加载器如下：

-   dc.molnet.load_delaney: V1. 在原始的论文中该数据集命名为 ESOL 数据集。
-   dc.molnet.load_sampl: V1. 在原始的论文中该数据集命名为  FreeSolv 数据集。
-   dc.molnet.load_lipo: V1. 在原始的论文中该数据集命名为  Lipophilicity in the original paper.
-   dc.molnet.load_thermosol: V2.
-   dc.molnet.load_hppb: V2.
-   dc.molnet.load_hopv: V2. 这个数据集来源于最近的 `这篇论文 <https://www.nature.com/articles/sdata201686>`_ 


化学反应数据集
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

这些数据集包含用于计算逆向合成/正向合成的化学反应数据集。 
相应的数据加载器如下：

-  dc.molnet.load_uspto

生化、生物物理数据集
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

这些数据集来自各种生化/生物物理数据集，这些数据集包含的性质诸如化合物与蛋白质的结合亲和力之类的东西。 
相应的数据加载器如下：

-    dc.molnet.load_pcba: V1
-    dc.molnet.load_nci: V2.
-    dc.molnet.load_muv: V1
-    dc.molnet.load_hiv: V1
-    dc.molnet.load_ppb: V2.
-    dc.molnet.load_bace_classification: V1. 
-   dc.molnet.load_bace_regression: V1. 
-   dc.molnet.load_kaggle: V2. 这个数据集来自于 Merck's drug discovery kaggle 竞赛，在这篇 `论文 <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.7b00146>`_  中有描述。
-   dc.molnet.load_factors: V2. 这个数据集来自于这篇 `论文 <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.7b00146>`_ 。 
-   dc.molnet.load_uv: V2.  这个数据集来自于这篇`论文 <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.7b00146>`_ 。
-   dc.molnet.load_kinase: V2. 这个数据集来自于这篇`论文 <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.7b00146>`_ 。




分子目录数据集 
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

这些数据集提供的分子数据集只包含了原始 SMILES 或者 结构的信息。 
这些类型的数据集对于分子生成建模任务很有用。 


-  dc.molnet.load_zinc15: V2
-  dc.molnet.load_chembl: V2
-  dc.molnet.load_chembl25: V2


生理学(Physiology)数据集
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

这些数据集测量分子如何与人类患者相互作用的生理特性。 

-  dc.molnet.load_bbbp: V1
-  dc.molnet.load_tox21: V1
-  dc.molnet.load_toxcast: V1
-  dc.molnet.load_sider: V1
-  dc.molnet.load_clintox: V1
-  dc.molnet.load_clearance: V2.



结构生物数据集
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

这些数据集包含大分子的 3D 结构以及相关属性。

-   dc.molnet.load_pdbbind: V1




显微镜数据集
$$$$$$$$$$$$$$$$$$$$$$$$$
这些数据集包含显微镜图像数据集，通常是细胞系。
 这些数据集不在最初的 MoleculeNet 论文中。

-   dc.molnet.load_bbbc001: V2
-   dc.molnet.load_bbbc002: V2
-   dc.molnet.load_cell_counting: V2


材料属性数据集 
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

这些数据集计算各种材料的属性。

-  dc.molnet.load_bandgap: V2
-  dc.molnet.load_perovskite: V2
-  dc.molnet.load_mp_formation_energy: V2
-  dc.molnet.load_mp_metallicity: V2


MoleculeNet 加载器解释
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

所有 MoleculeNet 加载器函数都采用 dc.molnet.load_X 的形式。 
加载器函数返回一组参数（任务tasks、数据集datasets、转换器transformers）。
现在让我们来深入了解一下这些返回值： 

-  tasks: 这是一个任务名称列表。 MoleculeNet 中的许多数据集都是“多任务”的。 也就是说，一个给定的数据点有多个与之关联的标签（属性）。 
-  datasets: 该字段是由三个 dc.data.Dataset 对象（train、valid、test）的元组。 这些对应于此 MoleculeNet 数据集的训练、验证和测试集。 
-  transformers: 此字段是dc.trans.Transformer对象的列表，应用于数据集的处理过程。 

上述的介绍可能有点抽象，以dc.molnet.load_delaney加载器函数为例， 让我们具体看看这3个变量是什么。


.. code-block:: python 

    import deepchem as dc 
    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv', splitter='random')
    train, valid, test = datasets

我们先来看看 tasks 变量，

.. code-block:: python 

    print(tasks)

输出 

.. code-block:: console 

    ['measured log solubility in mols per litre']

我们在此数据集中有一个任务，该任务是预测分子的溶解度的log数值。

再让我们来看看数据集：

.. code-block:: python 

    print(datasets)

输出：

.. code-block:: console 

    (<DiskDataset X.shape: (902,), y.shape: (902, 1), w.shape: (902, 1), ids: ['Clc1ccc(Cl)c(Cl)c1Cl' 'CCN(CC)C(=S)SSC(=S)N(CC)CC'
    'Nc3ccc2cc1ccccc1cc2c3 ' ... 'O=C1NC(=O)c2ccccc12 ' 'Clc1cccc(Cl)c1Cl'
    'O=C3CN=C(c1ccccc1)c2cc(ccc2N3)N(=O)=O'], task_names: ['measured log solubility in mols per litre']>, <DiskDataset X.shape: (113,), y.shape: (113, 1), w.shape: (113, 1), ids: ['C/C1CCC(\\C)CC1' 'c1cc2ccc3cccc4ccc(c1)c2c34' 'O=N(=O)c1ccccc1N(=O)=O'
    ... 'O=C1CCC(=O)N1' 'CCCC/C=C/C' 'CCN2c1cc(OC)cc(C)c1NC(=O)c3cccnc23 '], task_names: ['measured log solubility in mols per litre']>, <DiskDataset X.shape: (113,), y.shape: (113, 1), w.shape: (113, 1), ids: ['Brc1ccccc1Br' 'CCOC(=O)c1ccc(O)cc1' 'CCC(CC)C=O' ...
    'CCOc1ccc(cc1)C(C)(C)COCc3cccc(Oc2ccccc2)c3'
    'CC(=O)C1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C' 'C=Cc1ccccc1'], task_names: ['measured log solubility in mols per litre']>)

正如我们之前提到的，我们看到datasets是由3个数据集组成的元组。
让我们来看看这3个数据集，train,valid,test。

先看看训练集。
.. code-block:: python 

    print(train)


输出：

.. code-block:: console 

    <DiskDataset X.shape: (902,), y.shape: (902, 1), w.shape: (902, 1), ids: ['Clc1ccc(Cl)c(Cl)c1Cl' 'CCN(CC)C(=S)SSC(=S)N(CC)CC'
    'Nc3ccc2cc1ccccc1cc2c3 ' ... 'O=C1NC(=O)c2ccccc12 ' 'Clc1cccc(Cl)c1Cl'
    'O=C3CN=C(c1ccccc1)c2cc(ccc2N3)N(=O)=O'], task_names: ['measured log solubility in mols per litre']>

再看看验证集。
.. code-block:: python 

    print(valid)

输出： 

.. code-block:: console 

    <DiskDataset X.shape: (113,), y.shape: (113, 1), w.shape: (113, 1), ids: ['C/C1CCC(\\C)CC1' 'c1cc2ccc3cccc4ccc(c1)c2c34' 'O=N(=O)c1ccccc1N(=O)=O'
    ... 'O=C1CCC(=O)N1' 'CCCC/C=C/C' 'CCN2c1cc(OC)cc(C)c1NC(=O)c3cccnc23 '], task_names: ['measured log solubility in mols per litre']>

最后看看测试集。

.. code-block:: python 

    print(test)

输出：

.. code-block:: console 

    <DiskDataset X.shape: (113,), y.shape: (113, 1), w.shape: (113, 1), ids: ['Brc1ccccc1Br' 'CCOC(=O)c1ccc(O)cc1' 'CCC(CC)C=O' ...
    'CCOc1ccc(cc1)C(C)(C)COCc3cccc(Oc2ccccc2)c3'
    'CC(=O)C1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C' 'C=Cc1ccccc1'], task_names: ['measured log solubility in mols per litre']>

接下来让我们看看训练集中的一个样本数据。

.. code-block:: python 

    print(train.X[0])
    print(train.y[0])

输出：

.. code-block:: console 

    <deepchem.feat.mol_graphs.ConvMol object at 0x00000124A5213488>
    [-0.71096158]

Note that this is a dc.feat.mol_graphs.ConvMol object produced by dc.feat.ConvMolFeaturizer. We'll say more about how to control choice of featurization shortly. Finally let's take a look at the transformers field:

请注意，dc.feat.ConvMolFeaturizer 对象由 dc.feat.mol_graphs.ConvMol 函数产生。
稍后我们将详细如何选择制特征化。

 最后让我们看看transformers变量： 

.. code-block:: python 

    print(train.X[0])
    print(train.y[0])

输出：

.. code-block:: console 

    [<deepchem.trans.transformers.NormalizationTransformer object at 0x00000124A52F6CC8>]

我们看到了一个转换器 dc.trans.NormalizationTransformer。
到目前为止，您可能想知道在幕后做了哪些选择。
正如我们之前简提到的，可以使用不同选择的“特征器”来处理数据集。
我们可以在这里控制特征化的选择吗？
另外，源数据集是如何分成训练/有效/测试三个不同的数据集的？

您可以使用 'featurizer' 和 'splitter' 关键字参数并传入不同的字符串。 
'featurizer' 的常见可能选择是 'ECFP'、'GraphConv'、'Weave' 和 'smiles2img' 对应于 dc.feat.CircularFingerprint、dc.feat.ConvMolFeaturizer、dc.feat.WeaveFeaturizer 和 dc.feat.SmilesToImage 特征化器. 
'splitter' 的常见可能选择是 None、'index'、'random'、'scaffold' 和 'stratified' 对应于no split、dc.splits.IndexSplitter、dc.splits.RandomSplitter、dc.splits.SingletaskStratifiedSplitter。




你可以传入任何 featurizer 或 splitter 参数给dc.molnet.load_delaney函数。

.. code-block:: python 

    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer="ECFP", splitter="scaffold")
    (train, valid, test) = datasets
    print(train.X[0])

输出：

.. code-block:: console 

    [0. 0. 0. ... 0. 0. 0.]

请注意，与之前的调用使用的特征函数不同，这次我们使用 dc.feat.CircularFingerprint 函数生成的 numpy 数组作为特征，而不是 dc.feat.ConvMolFeaturizer 生成的 ConvMol 对象作为特征。 



自己试试，尝试调用 MoleculeNet 来加载一些其他数据集并尝试使用不同的特征化器/拆分参数，看看会发生什么！ 



