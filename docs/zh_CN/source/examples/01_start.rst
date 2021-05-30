DeepChem快速入门教程
======================================
欢迎来到DeepChem的介绍教程。
我们将通过一系列的教程让你更全面的了解这个工具，
从而帮助你更好的将深度学习技术应用到生命科学领域。
如果你是第一次接触这个工具，建议先看入门教程1。



入门教程
----------------

.. contents:: 目录
    :local:


安装DeepChem
^^^^^^^^^^^^^^^^
首先，安装 `anaconda3 <https://www.anaconda.com/products/individual>`_。

然后，创建deepchem的anaconda环境。

.. code-block:: bash

    conda create --name py37deepchem python=3.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main


最后，安装deepchem

.. code-block:: bash

    conda activate py37deepchem
    pip install tensorflow~=2.4  -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
    pip install deepchem -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
    conda install rdkit -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/rdkit/
    conda install numpy=1.19.5 -c conda-forge

.. note::
	
	注意 tensorflow和pytorch这样的高级框架，对于相关库的版本和环境是有严格要求的，建议使用 1.19.5 版本的numpy。

查看deepchem的版本

.. code-block:: python 

    import deepchem as dc 
    dc.__version__ 


使用DeepChem训练模型：第一个案例
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
深度学习可以用于解决各种各样的问题，但是基本流程是一样的。下面是一些常见的操作步骤。

1. 选择用于训练的数据集（如果没有合适的数据集，自己创建一个数据集）；
2. 创建模型；
3. 在数据集上训练模型； Train the model on the data.
4. 在独立的测试集上评价模型。
5. 使用模型进行预测



使用DeepChem，上述流程中的每一个步骤都只需一两行Python代码。 
在本教程中，我们将通过一个基本示例展示如何借助DeepChem解决现实世界中的科学问题。 

我们将解决的问题是根据小分子的smiles预测小分子的溶解度。 
这是药物开发中一个非常重要的特性：如果设计的药物分子的溶解度不够，那么药物分子可能难以进入患者的血液并产生治疗效果。 
我们需要的第一件事是真实分子的溶解度数据集。 DeepChem 的核心组件之一是 MoleculeNet，这是一个多样化的化学和分子数据集集合。
对于本教程，我们可以使用 Delaney 溶解度数据集。 


首先载入数据集, 设置数据集的特征方式（GraphConv），分割数据集为训练集、验证集、测试集。

.. code-block:: python 

    import deepchem as dc 
    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv')
    train_dataset, valid_dataset, test_dataset = datasets

不在这里对这段代码进行详细解释。 我们将在后面的教程中看到许多类似的例子。 注意两个细节。 
首先，注意传递给:code:`load_delaney()`函数的 featurizer 参数， 分子可以用多种方式表示。 
因此，我们告诉它我们想要使用哪种表示，或者用更专业的语言，告诉它如何“特征化”数据。
其次，注意这里得到了三个不同的数据集：训练集、验证集和测试集。 在深度学习工作流程中，每一个数据集都有其特定的作用。


现在我们有了数据，下一步是创建模型。 我们将使用一种特定的模型，称为“图卷积网络（graph convolutional network）”，简称为“ graphconv”。 

.. code-block:: python 

    model = dc.models.GraphConvModel(n_tasks=1, mode='regression', dropout=0.2)

在这里我不会对上述代码进行详细解释。 后面的教程将提供有关 GraphConvModel等其他模型的详细信息。 

我们现在需要在数据集上训练模型。 我们只是给它数据集并告诉它要执行多少个训练周期（epoch）（即，要完成多少次完整的数据传递）。

.. code-block:: python 

    model.fit(train_dataset, nb_epoch=100)


如果一切顺利，我们现在应该有一个经过完全训练的模型！ 
为了验证模型的预测能力，我们必须在测试集上评估模型。 
我们通过选择一个评估指标并在模型上调用 :code:`evaluate()`函数来评估模型的预测能力。 
对于此示例，让我们使用 Pearson 相关性（也称为 r**2）作为我们的指标。 
我们可以在训练集和测试集上对其进行评估。 

:durole:`superscript`

.. code-block:: python 

    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
    print("Training set score:", model.evaluate(train_dataset, [metric], transformers))
    print("Test set score:", model.evaluate(test_dataset, [metric], transformers))


输出：

.. code-block:: console

    Training set score: {'pearson_r2_score': 0.9181928383940342}
    Test set score: {'pearson_r2_score': 0.663163746029648}


我们发现模型在训练集上的得分高于测试集。 
与在相似但独立的数据上相比，模型在训练的特定数据上的表现通常更好。 
这就是所谓的“过度拟合”，也是需要在独立的测试集上评估模型至关重要的原因。
我们的模型在测试集上仍然具有一定的预测能力。 产生完全随机输出的模型的相关性为 0，而做出完美预测的模型的相关性为 1。
我们的模型有一定的预测能力，所以现在我们可以用它来预测我们关心的其他分子的溶解度性质。 

由于这只是一个教程，我们没有特别想要预测的任何其他分子，我们对测试集中的前十个分子进行预测。 
对于每一个分子，我们打印出分子的SMILES字符串、实际溶解度和预测溶解度。


.. code-block:: python 

    solubilities = model.predict_on_batch(test_dataset.X[:10])
    for molecule, solubility, test_solubility in zip(test_dataset.ids, solubilities, test_dataset.y):
        print(molecule,solubility, test_solubility)

输出：

.. code-block:: console

    c1cc2ccc3cccc4ccc(c1)c2c34 [-1.6963764] [-1.60114461]
    Cc1cc(=O)[nH]c(=S)[nH]1 [0.7654593] [0.20848251]
    Oc1ccc(cc1)C2(OC(=O)c3ccccc23)c4ccc(O)cc4  [-0.26484838] [-0.01602738]
    c1ccc2c(c1)cc3ccc4cccc5ccc2c3c45 [-1.8617188] [-2.82191713]
    C1=Cc2cccc3cccc1c23 [-1.1605877] [-0.52891635]
    CC1CO1 [1.3871247] [1.10168349]
    CCN2c1ccccc1N(C)C(=S)c3cccnc23  [-0.08044883] [-0.88987406]
    CC12CCC3C(CCc4cc(O)ccc34)C2CCC1=O [-0.5294326] [-0.52649706]
    Cn2cc(c1ccccc1)c(=O)c(c2)c3cccc(c3)C(F)(F)F [-0.78735524] [-0.76358725]
    ClC(Cl)(Cl)C(NC=O)N1C=CN(C=C1)C(NC=O)C(Cl)(Cl)Cl  [-0.36010832] [-0.64020358]



完整代码
^^^^^^^^^^^^^^^^^^^^^^
将下面的代码，保存为文件 01start_dc.py。 

.. code-block:: python 

    import deepchem as dc 
    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv')
    train_dataset, valid_dataset, test_dataset = datasets
    model = dc.models.GraphConvModel(n_tasks=1, mode='regression', dropout=0.2)
    model.fit(train_dataset, nb_epoch=100)
    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
    print("Training set score:", model.evaluate(train_dataset, [metric], transformers))
    print("Test set score:", model.evaluate(test_dataset, [metric], transformers))

在py37deepchem的环境下运行上述脚本

.. code-block:: bash 

    python  01start_dc.py 



恭喜！欢迎加入DeepChem社区
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^






恭喜！欢迎加入DeepChem社区
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
恭喜您完成本教程！ 
如果您喜欢完成本教程，并希望继续使用 DeepChem，
我们鼓励您完成本系列中的其余教程。
您还可以通过以下方式帮助 DeepChem 社区： 

1. 为github上面的`DeepChem <https://github.com/deepchem/deepchem>`点赞 ;

这有助于提高DeepChem项目和工具在药物发现社区中的关注度。



2. 加入DeepChem的`Gitter社区 <https://gitter.im/deepchem/Lobby>`_

DeepChem Gitter 聚集了许多对生命科学领域的深度学习感兴趣的科学家、开发人员和爱好者。 加入和他们一起交流吧。




    


