DeepChem教程2：Datasets的相关操作
======================================
欢迎来到DeepChem的介绍教程。
我们将通过一系列的教程让你更全面的了解这个工具，
从而帮助你更好的将深度学习技术应用到生命科学领域。
如果你是第一次接触这个工具，
建议先看 `入门教程1 <https://deepchembook.readthedocs.io/zh_CN/latest/examples/tutorials/01_start.html>`_。



数据集（Dataset）的操作
---------------------------

.. contents:: 目录
    :local:



数据集（ Dataset）的解读
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
在`上一个教程 <https://deepchembook.readthedocs.io/zh_CN/latest/examples/tutorials/01_start.html>`_ 中，
我们加载了Delaney 数据集中的分子溶解度数据。 现在我们重新加载它。

.. code-block:: python 

    import deepchem as dc 
    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv')
    train_dataset, valid_dataset, test_dataset = datasets

现在我们有3个数据集：训练集、验证集和测试集。这些数据集中包含哪些信息？
我们可以通过打印输出这些变量，来了解这些数据集的内容。

.. code-block:: python 

    print("test_dataset type is",type(test_dataset))
    print("test_dataset properties are ",dir(test_dataset))
    print("test_dataset is", test_dataset)

输出：

.. code-block:: console 

    test_dataset type is <class 'deepchem.data.datasets.DiskDataset'>
    test_dataset properties are  ['X', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_cache_used', '_cached_shards', '_construct_metadata', '_get_metadata_filename', '_get_shard_shape', '_iterbatches_from_shards', '_memory_cache_size', '_save_metadata', '_transform_shard', 'add_shard', 'complete_shuffle', 'copy', 'create_dataset', 'data_dir', 'from_dataframe', 'from_numpy', 'get_data_shape', 'get_label_means', 'get_label_stds', 'get_number_shards', 'get_shape', 'get_shard', 'get_shard_ids', 'get_shard_size', 'get_shard_w', 'get_shard_y', 'get_statistics', 'get_task_names', 'ids', 'iterbatches', 'itersamples', 'itershards', 'legacy_metadata', 'load_metadata', 'make_pytorch_dataset', 'make_tf_dataset', 'memory_cache_size', 'merge', 'metadata_df', 'move', 'reshard', 'save_to_disk', 'select', 'set_shard', 'shuffle_each_shard', 'shuffle_shards', 'sparse_shuffle', 'subset', 'tasks', 'to_dataframe', 'transform', 'w', 'write_data_to_disk', 'y']
    test_dataset is <DiskDataset X.shape: (113,), y.shape: (113, 1), w.shape: (113, 1), ids: ['c1cc2ccc3cccc4ccc(c1)c2c34' 'Cc1cc(=O)[nH]c(=S)[nH]1'
    'Oc1ccc(cc1)C2(OC(=O)c3ccccc23)c4ccc(O)cc4 ' ...
    'c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc43' 'Cc1occc1C(=O)Nc2ccccc2'
    'OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O '], task_names: ['measured log solubility in mols per litre']>

这里有很多信息，所以让我们从头开始。
我们可以看到变量test_dataset变量的类型是 :code:`deepchem.data.datasets.DiskDataset` 类。这个类中有各种各样的方法和属性。

"DiskDataset"是 "datasets"的一个子类。datasets 一共有3个子类：

- DiskDataset 是已保存到磁盘的数据集。 即使数据总量远远大于计算机内存，也可以有效访问的方式进行访问。
- NumpyDataset 是一个内存数据集，包含 NumPy 数组中的所有数据。 在处理完全适合内存的中小型数据集时，该类非常适合
- ImageDataset 是一个更专业的类，它能存储在磁盘上的图像文件的部分或者全部数据。 当使用以图像作为输入或输出的模型时，该类非常有用。 

现在让我们看看数据集的内容。 每个数据集存储一系列的样本列表。 
简单地说，样本是单个数据点。 
在这种案例中，每个样本都是一个分子及其信息。 
在其他数据集中，样本可能对应于实验测定、细胞系、图像或许多其他事物。 
对于每个样本，数据集存储以下信息：

- 特征信息，通常写作大小的X。 对样本进行特征化，作为模型的输入信息。 
- 标签信息，通常写作小写的y。期待模型的输出信息。在训练过程，通过对模型的参数进行优化使得模型的输出尽可能接近y。
- 权重，通常写作小写的w。它可以用于表征数据的中重要程度。在后面的教程中，我们会展示该信息的重要性。
- ID, 每个样本都有一个独特的ID编号。这可以是任何东西，只要它是独一无二的。 有时它只是一个整数索引，但在这个数据集中，ID 是一个描述分子的 SMILES 字符串。 
- task_names,输出中列出的最后一条信息是 task_names。
请注意，X，y和w的第一维的大小均为113。 这意味着该数据集包含 113 个样本。 

一些数据集包含每个样本的多条信息。例如，如果一个样本代表一个分子，数据集可能会记录对该分子的多个不同实验的结果。
这个数据集仅仅记录了分子的溶解度信息。

注意 y 和 w 的维度是 (113, 1)。第一维的大小代表 这些数组的第二维通常与任务数相匹配。
对于该案例，只有一个任务预测溶解度，因此第二维的大小为1。




从 Dataset中访问数据
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

有多种方法可以访问数据集中包含的数据。 最简单的就是直接访问 X、y、w 和 ids 属性，以 NumPy 数组的形式返回相应的信息。 

比如访问数据集中的样本的y信息

.. code-block:: python 

    print("y in test_dataset",test_dataset.y)
  
.. code-block:: console 

    y in test_dataset [[-1.60114461]
    [ 0.20848251]
    [-0.01602738]
    [-2.82191713]
    [-0.52891635]
    [ 1.10168349]
    [-0.88987406]
    [-0.52649706]
    [-0.76358725]
    [-0.64020358]
    [-0.38569452]
    [-0.62568785]
    [-0.39585553]
    [-2.05306753]
    [-0.29666474]
    [-0.73213651]
    [-1.27744393]
    [ 0.0081655 ]
    [ 0.97588054]
    [-0.10796031]
    [ 0.59847167]
    [-0.60149498]
    [-0.34988907]
    [ 0.34686576]
    [ 0.62750312]
    [ 0.14848418]
    [ 0.02268122]
    [-0.85310089]
    [-2.72079091]
    [ 0.42476682]
    [ 0.01300407]
    [-2.4851523 ]
    [-2.15516147]
    [ 1.00975056]
    [ 0.82588471]
    [-0.90390593]
    [-0.91067993]
    [-0.82455329]
    [ 1.26909819]
    [-1.14825397]
    [-2.1343556 ]
    [-1.15744727]
    [-0.1045733 ]
    [ 0.53073162]
    [-1.22567118]
    [-1.66452995]
    [ 0.24525568]
    [-0.13215318]
    [-0.97067826]
    [-0.23376326]
    [ 1.21297072]
    [-1.2595412 ]
    [ 0.49686159]
    [ 0.22396595]
    [-0.44182199]
    [ 0.47895886]
    [ 0.08267956]
    [-1.51840498]
    [-0.34795364]
    [-0.83858516]
    [-0.13699176]
    [-2.59498796]
    [ 0.13106531]
    [ 0.09042128]
    [ 1.18877785]
    [-0.82697258]
    [-1.16857599]
    [ 0.37589721]
    [-0.24344041]
    [-2.00952036]
    [-0.59181783]
    [-0.15634606]
    [-2.87272217]
    [-0.34069577]
    [ 0.27622256]
    [-2.15467761]
    [-0.02812382]
    [-2.77401524]
    [ 0.25638441]
    [ 0.84040043]
    [-0.86277804]
    [-1.52082426]
    [ 0.29702844]
    [ 0.44363727]
    [ 0.47460415]
    [-0.08376743]
    [ 0.68556602]
    [ 0.79201468]
    [-1.2401869 ]
    [ 0.6129874 ]
    [-0.58214068]
    [-1.51598569]
    [-1.93984487]
    [-0.30295489]
    [-0.24827899]
    [ 1.06442646]
    [-1.48259952]
    [ 0.0275198 ]
    [ 0.33718861]
    [-0.91600236]
    [ 0.58637523]
    [-0.62084928]
    [-0.30827732]
    [-1.95145746]
    [-0.83568202]
    [ 0.10977558]
    [ 1.90488697]
    [-0.75149081]
    [-1.65630437]
    [ 0.74362893]
    [-2.42079925]
    [-0.20957039]
    [ 1.01458914]]

这是访问数据的一种非常简单的方法，但您在使用它时应该非常小心。 这需要一次将所有样本的数据加载到内存中。
 这对于像这样的小数据集来说没有问题，但对于大型数据集，容易因内存不足而使程序崩溃。 

更好的方法访问数据的方式是迭代数据集。 让它一次只加载一点数据，处理它，然后在加载下一位之前释放内存。 
您可以使用 itersamples() 方法一次迭代一个样本。 

.. code-block:: python 

    for X, y, w, id in test_dataset.itersamples():
        print(y, id)


输出：
.. code-block:: console 

    [-1.60114461] c1cc2ccc3cccc4ccc(c1)c2c34
    [0.20848251] Cc1cc(=O)[nH]c(=S)[nH]1
    [-0.01602738] Oc1ccc(cc1)C2(OC(=O)c3ccccc23)c4ccc(O)cc4
    [-2.82191713] c1ccc2c(c1)cc3ccc4cccc5ccc2c3c45
    [-0.52891635] C1=Cc2cccc3cccc1c23
    [1.10168349] CC1CO1
    [-0.88987406] CCN2c1ccccc1N(C)C(=S)c3cccnc23
    [-0.52649706] CC12CCC3C(CCc4cc(O)ccc34)C2CCC1=O
    [-0.76358725] Cn2cc(c1ccccc1)c(=O)c(c2)c3cccc(c3)C(F)(F)F
    [-0.64020358] ClC(Cl)(Cl)C(NC=O)N1C=CN(C=C1)C(NC=O)C(Cl)(Cl)Cl
    [-0.38569452] COc2c1occc1cc3ccc(=O)oc23
    [-0.62568785] CN2C(=C(O)c1ccccc1S2(=O)=O)C(=O)Nc3ccccn3
    [-0.39585553] Cc3cc2nc1c(=O)[nH]c(=O)nc1n(CC(O)C(O)C(O)CO)c2cc3C
    [-2.05306753] c1ccc(cc1)c2ccc(cc2)c3ccccc3
    [-0.29666474] CC34CC(=O)C1C(CCC2=CC(=O)CCC12C)C3CCC4(=O)
    [-0.73213651] c1ccc2c(c1)sc3ccccc23
    [-1.27744393] CC23Cc1cnoc1C=C2CCC4C3CCC5(C)C4CCC5(O)C#C
    [0.0081655] OC(C(=O)c1ccccc1)c2ccccc2
    [0.97588054] OCC2OC(Oc1ccccc1CO)C(O)C(O)C2O
    [-0.10796031] CC3C2CCC1(C)C=CC(=O)C(=C1C2OC3=O)C
    [0.59847167] O=Cc2ccc1OCOc1c2
    [-0.60149498] CC1CCCCC1NC(=O)Nc2ccccc2
    [-0.34988907] CC(=O)N(S(=O)c1ccc(N)cc1)c2onc(C)c2C
    [0.34686576] C1N(C(=O)NCC(C)C)C(=O)NC1
    [0.62750312] CNC(=O)Oc1ccccc1C2OCCO2
    [0.14848418] CC1=C(CCCO1)C(=O)Nc2ccccc2
    [0.02268122] Cn2c(=O)on(c1ccc(Cl)c(Cl)c1)c2=O
    [-0.85310089] C1Cc2cccc3cccc1c23
    [-2.72079091] c1ccc2cc3c4cccc5cccc(c3cc2c1)c45
    [0.42476682] Nc1cc(nc(N)n1=O)N2CCCCC2
    [0.01300407] O=c2c(C3CCCc4ccccc43)c(O)c1ccccc1o2
    [-2.4851523] CC(C)C(Nc1ccc(cc1Cl)C(F)(F)F)C(=O)OC(C#N)c2cccc(Oc3ccccc3)c2
    [-2.15516147] Cc1c(F)c(F)c(COC(=O)C2C(C=C(Cl)C(F)(F)F)C2(C)C)c(F)c1F
    [1.00975056] c2ccc1[nH]nnc1c2
    [0.82588471] c2ccc1ocnc1c2
    [-0.90390593] CCOC(=O)c1cncn1C(C)c2ccccc2
    [-0.91067993] CCN2c1ccccc1N(C)C(=O)c3ccccc23
    [-0.82455329] OCC(O)COC(=O)c1ccccc1Nc2ccnc3cc(Cl)ccc23
    [1.26909819] OCC1OC(OC2C(O)C(O)C(O)OC2CO)C(O)C(O)C1O
    [-1.14825397] CC34CCc1c(ccc2cc(O)ccc12)C3CCC4=O
    [-2.1343556] ClC1=C(Cl)C(Cl)(C(=C1Cl)Cl)C2(Cl)C(=C(Cl)C(=C2Cl)Cl)Cl
    [-1.15744727] ClC1(C(=O)C2(Cl)C3(Cl)C14Cl)C5(Cl)C2(Cl)C3(Cl)C(Cl)(Cl)C45Cl
    [-0.1045733] Oc1ccc(c(O)c1)c3oc2cc(O)cc(O)c2c(=O)c3O
    [0.53073162] C1SC(=S)NC1(=O)
    [-1.22567118] ClC(Cl)C(Cl)(Cl)SN2C(=O)C1CC=CCC1C2=O
    [-1.66452995] ClC1=C(Cl)C2(Cl)C3C4CC(C=C4)C3C1(Cl)C2(Cl)Cl
    [0.24525568] CC(=O)Nc1nnc(s1)S(N)(=O)=O
    [-0.13215318] CC1=C(SCCO1)C(=O)Nc2ccccc2
    [-0.97067826] CN(C(=O)COc1nc2ccccc2s1)c3ccccc3
    [-0.23376326] CN(C(=O)NC(C)(C)c1ccccc1)c2ccccc2
    [1.21297072] Nc1nccs1
    [-1.2595412] CN(C=Nc1ccc(C)cc1C)C=Nc2ccc(C)cc2C
    [0.49686159] OCC(O)C2OC1OC(OC1C2O)C(Cl)(Cl)Cl
    [0.22396595] Nc3nc(N)c2nc(c1ccccc1)c(N)nc2n3
    [-0.44182199] CC2Nc1cc(Cl)c(cc1C(=O)N2c3ccccc3C)S(N)(=O)=O
    [0.47895886] CN1CC(O)N(C1=O)c2nnc(s2)C(C)(C)C
    [0.08267956] CCC1(C(=O)NC(=O)NC1=O)C2=CCC3CCC2C3
    [-1.51840498] CCC(C)C(=O)OC2CC(C)C=C3C=CC(C)C(CCC1CC(O)CC(=O)O1)C23
    [-0.34795364] CC2Cc1ccccc1N2NC(=O)c3ccc(Cl)c(c3)S(N)(=O)=O
    [-0.83858516] o1c2ccccc2c3ccccc13
    [-0.13699176] O=C(Nc1ccccc1)Nc2ccccc2
    [-2.59498796] c1ccc2c(c1)c3cccc4c3c2cc5ccccc54
    [0.13106531] COc1ccc(cc1)C(O)(C2CC2)c3cncnc3
    [0.09042128] c1cnc2c(c1)ccc3ncccc23
    [1.18877785] OCC1OC(CO)(OC2OC(COC3OC(CO)C(O)C(O)C3O)C(O)C(O)C2O)C(O)C1O
    [-0.82697258] CCOC(=O)c1ccccc1S(=O)(=O)NN(C=O)c2nc(Cl)cc(OC)n2
    [-1.16857599] CC34CCC1C(=CCc2cc(O)ccc12)C3CCC4=O
    [0.37589721] CN(C)C(=O)Oc1cc(C)nn1c2ccccc2
    [-0.24344041] OC(Cn1cncn1)(c2ccc(F)cc2)c3ccccc3F
    [-2.00952036] Cc1c2ccccc2c(C)c3ccc4ccccc4c13
    [-0.59181783] Cc3nnc4CN=C(c1ccccc1Cl)c2cc(Cl)ccc2n34
    [-0.15634606] Cc3ccnc4N(C1CC1)c2ncccc2C(=O)Nc34
    [-2.87272217] c1cc2cccc3c4cccc5cccc(c(c1)c23)c54
    [-0.34069577] COc1cc(cc(OC)c1O)C6C2C(COC2=O)C(OC4OC3COC(C)OC3C(O)C4O)c7cc5OCOc5cc67
    [0.27622256] O=c1[nH]cnc2nc[nH]c12
    [-2.15467761] C1C(O)CCC2(C)CC3CCC4(C)C5(C)CC6OCC(C)CC6OC5CC4C3C=C21
    [-0.02812382] Cc1ccccc1n3c(C)nc2ccccc2c3=O
    [-2.77401524] CCOc1ccc(cc1)C(C)(C)COCc3cccc(Oc2ccccc2)c3
    [0.25638441] CCC1(CCC(=O)NC1=O)c2ccccc2
    [0.84040043] CC1CC(C)C(=O)C(C1)C(O)CC2CC(=O)NC(=O)C2
    [-0.86277804] CC(=O)C3CCC4C2CC=C1CC(O)CCC1(C)C2CCC34C
    [-1.52082426] Cc1ccc(OP(=O)(Oc2cccc(C)c2)Oc3ccccc3C)cc1
    [0.29702844] CSc1nnc(c(=O)n1N)C(C)(C)C
    [0.44363727] Nc1ncnc2n(ccc12)C3OC(CO)C(O)C3O
    [0.47460415] O=C2NC(=O)C1(CC1)C(=O)N2
    [-0.08376743] C1Cc2ccccc2C1
    [0.68556602] c1ccc2cnccc2c1
    [0.79201468] OCC1OC(C(O)C1O)n2cnc3c(O)ncnc23
    [-1.2401869] c2(Cl)c(Cl)c(Cl)c1nccnc1c2(Cl)
    [0.6129874] C1OC1c2ccccc2
    [-0.58214068] CCC(=C(CC)c1ccc(O)cc1)c2ccc(O)cc2
    [-1.51598569] c1ccc2c(c1)c3cccc4cccc2c34
    [-1.93984487] CC(C)C(C(=O)OC(C#N)c1cccc(Oc2ccccc2)c1)c3ccc(OC(F)F)cc3
    [-0.30295489] CCCC1COC(Cn2cncn2)(O1)c3ccc(Cl)cc3Cl
    [-0.24827899] O=C2CN(N=Cc1ccc(o1)N(=O)=O)C(=O)N2
    [1.06442646] NC(=O)c1cnccn1
    [-1.48259952] OC4=C(C1CCC(CC1)c2ccc(Cl)cc2)C(=O)c3ccccc3C4=O
    [0.0275198] O=C(Cn1ccnc1N(=O)=O)NCc2ccccc2
    [0.33718861] CCC1(C(=O)NC(=O)NC1=O)C2=CCCCC2
    [-0.91600236] COC(=O)C1=C(C)NC(=C(C1c2ccccc2N(=O)=O)C(=O)OC)C
    [0.58637523] O=C2NC(=O)C1(CCC1)C(=O)N2
    [-0.62084928] CCCOP(=S)(OCCC)SCC(=O)N1CCCCC1C
    [-0.30827732] N(c1ccccc1)c2ccccc2
    [-1.95145746] ClC(Cl)=C(c1ccc(Cl)cc1)c2ccc(Cl)cc2
    [-0.83568202] O=c2[nH]c1CCCc1c(=O)n2C3CCCCC3
    [0.10977558] CCC1(C(=O)NCNC1=O)c2ccccc2
    [1.90488697] O=C1CCCN1
    [-0.75149081] COc5cc4OCC3Oc2c1CC(Oc1ccc2C(=O)C3c4cc5OC)C(C)=C
    [-1.65630437] ClC4=C(Cl)C5(Cl)C3C1CC(C2OC12)C3C4(Cl)C5(Cl)Cl
    [0.74362893] c1ccsc1
    [-2.42079925] c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc43
    [-0.20957039] Cc1occc1C(=O)Nc2ccccc2
    [1.01458914] OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O

大多数深度学习模型可以一次处理多个样本（batch）。 您可以使用 iterbatches() 迭代多个样本。 

.. code-block:: python 

    for X, y, w, ids in test_dataset.iterbatches(batch_size=50):
        print(y.shape)

输出：

.. code-block:: console 

    (50, 1)
    (50, 1)
    (13, 1)






在训练模型时，iterbatches() 有其他有用的功能。 
例如， iterbatches(batch_size=100, epochs=10, deterministic=False) 将在整个数据集上迭代十轮，每轮使用不同随机顺序的样本。



Datasets 还提供了使用 TensorFlow 和 PyTorch 的数据接口。 要获取一个tensorflow.data.Dataset，请调用make_tf_dataset（）函数。 
要获取torch.utils.data.IterableDataset，请调用make_pytorch_dataset（）。 

访问数据的最后一种方式是 to_dataframe()。 这会将数据转换成 Pandas DataFrame 的形式。
这需要一次将所有数据存储在内存中，因此您应该只将它用于小数据集。 

.. code-block:: python 

    test_dataset.to_dataframe()



输出：

.. code-block:: console 
                                                        X  ...                                                ids
    0    <deepchem.feat.mol_graphs.ConvMol object at 0x...  ...                         c1cc2ccc3cccc4ccc(c1)c2c34
    1    <deepchem.feat.mol_graphs.ConvMol object at 0x...  ...                            Cc1cc(=O)[nH]c(=S)[nH]1
    2    <deepchem.feat.mol_graphs.ConvMol object at 0x...  ...         Oc1ccc(cc1)C2(OC(=O)c3ccccc23)c4ccc(O)cc4
    3    <deepchem.feat.mol_graphs.ConvMol object at 0x...  ...                   c1ccc2c(c1)cc3ccc4cccc5ccc2c3c45
    4    <deepchem.feat.mol_graphs.ConvMol object at 0x...  ...                                C1=Cc2cccc3cccc1c23
    ..                                                 ...  ...                                                ...
    108  <deepchem.feat.mol_graphs.ConvMol object at 0x...  ...     ClC4=C(Cl)C5(Cl)C3C1CC(C2OC12)C3C4(Cl)C5(Cl)Cl
    109  <deepchem.feat.mol_graphs.ConvMol object at 0x...  ...                                            c1ccsc1
    110  <deepchem.feat.mol_graphs.ConvMol object at 0x...  ...                 c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc43
    111  <deepchem.feat.mol_graphs.ConvMol object at 0x...  ...                             Cc1occc1C(=O)Nc2ccccc2
    112  <deepchem.feat.mol_graphs.ConvMol object at 0x...  ...  OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)...



创建数据集 Datasets 
^^^^^^^^^^^^^^^^^^^^^^


谈谈如何创建自己的数据集。 创建NumpyDataset非常简单：只需将包含数据的数组传递给构造函数即可。 
让我们创建一些随机数组，然后将它们包装在 NumpyDataset 中。 

.. code-block:: python 

    import numpy as np

    X = np.random.random((10, 5))
    y = np.random.random((10, 2))
    dataset = dc.data.NumpyDataset(X=X, y=y)
    print(dataset)

输出：

.. code-block:: console 

    <NumpyDataset X.shape: (10, 5), y.shape: (10, 2), w.shape: (10, 1), ids: [0 1 2 3 4 5 6 7 8 9], task_names: [0 1]>

注意，我们没有指定权重w或 ID, 这些是可选的。 
只需要 X。 由于我们没有指定他们，它会自动为我们构建 w 和 id 数组，将所有权重设置为 1，并将 ID 设置为整数索引。

通过将其转换为 Pandas DataFrame 的形式，查看其具体的内容。

.. code-block:: python 

    print(dataset.to_dataframe())


.. code-block:: console 

            X1        X2        X3        X4        X5        y1        y2    w ids
    0  0.237623  0.885838  0.185449  0.041476  0.982166  0.028134  0.491598  1.0   0
    1  0.490529  0.017464  0.331176  0.142093  0.672005  0.267942  0.330839  1.0   1
    2  0.314899  0.415268  0.097622  0.417283  0.519209  0.241511  0.286500  1.0   2
    3  0.071865  0.589685  0.490738  0.355478  0.208175  0.007239  0.410269  1.0   3
    4  0.284844  0.745729  0.143815  0.144825  0.514067  0.546191  0.957701  1.0   4
    5  0.422026  0.453786  0.351375  0.981475  0.125982  0.488564  0.181026  1.0   5
    6  0.298952  0.418125  0.037490  0.005730  0.025157  0.090561  0.273588  1.0   6
    7  0.155246  0.928438  0.954274  0.281273  0.145900  0.313455  0.237399  1.0   7
    8  0.654904  0.158257  0.394742  0.934613  0.660716  0.995862  0.881379  1.0   8
    9  0.895681  0.504728  0.622640  0.349956  0.211222  0.653983  0.952951  1.0   9


如何创建一个 DiskDataset？ 如果您有 NumPy 数组中的数据，则可以调用 DiskDataset.from_numpy() 将其保存到磁盘中。 
由于这只是一个教程，我们将其保存到一个临时目录。 


.. code-block:: python 


    import tempfile
    with tempfile.TemporaryDirectory() as data_dir:
        disk_dataset = dc.data.DiskDataset.from_numpy(X=X, y=y, data_dir=data_dir)
        print(disk_dataset)



输出：

.. code-block:: console 

    <DiskDataset X.shape: (10, 5), y.shape: (10, 2), w.shape: (10, 1), ids: [0 1 2 3 4 5 6 7 8 9], task_names: [0 1]>



如果数据集很大，不能一次性载入内存中，如果创建DiskDataset？ 
如果磁盘上有一些包含数亿个分子数据的大文件怎么办？
基于这些创建 DiskDataset 的过程稍微复杂一些。 幸运的是，DeepChem 的 DataLoader 框架可以为您自动化大部分工作。 
这是一个非常重要的主题，因此我们将在后面的教程中进行详细说明。 






