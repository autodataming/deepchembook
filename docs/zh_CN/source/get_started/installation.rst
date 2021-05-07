安装
============

稳定版本
--------------


安装deepchem之前先安装tensorflow 2.4 版本。

笔者建议使用conda管理各种python项目的环境。


.. code-block:: bash

    conda create --name py37deepchem python=3.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    conda activate py37deepchem
    pip install tensorflow~=2.4
    pip install deepchem


RDKit 是一个可选的安装包， 推荐安装RDKit因为好多有用的方法都依赖它，比如 molnet 。


.. code-block:: bash

    conda install rdkit -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/rdkit/

最新测试版本
-------------------------

你也可以通过PIP安装deepchem的最新测试版本，该版本可能会有一些新功能，以及一些没有被发现的Bug。

.. code-block:: bash

    pip install tensorflow~=2.4
    pip install --pre deepchem







测试是否安装成功
-----------------------------------

.. code-block:: bash

    (deepchem) root@xxxxxxxxxxxxx:~/mydir# python
    Python 3.6.10 |Anaconda, Inc.| (default, May  8 2020, 02:54:21)
    [GCC 7.3.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import deepchem as dc


在tox21数据集上进行测试
-----------------------------------------
.. code-block:: bash

    wget https://raw.githubusercontent.com/deepchem/deepchem/master/examples/benchmark.py
    python benchmark.py -d tox21 -m graphconv -s random



