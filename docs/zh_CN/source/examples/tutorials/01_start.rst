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
    pip install tensorflow~=2.4
    pip install deepchem
    conda install rdkit -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/rdkit/


是否有 DeepChem 首席研究员?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
尽管DeepChem诞生于斯坦福大学的Pande 实验室，
 但是现在该项目以去中心研究组织的形式存在。
更准确的说，有多个非正式的"DeepChem PIs"，
只要你在工作中使用DeepChem,你就是一个DeepChem PI！
