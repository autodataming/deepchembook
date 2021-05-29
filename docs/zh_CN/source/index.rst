

.. deepchemCN documentation master file, created by
   sphinx-quickstart on Tue May  4 20:50:47 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎来到DeepChem中文文档教程
======================================


.. raw:: html

  <embed>
    <a href="https://github.com/autodataming/deepchembook">
      <img style="position: absolute; top: 0; right: 0; border: 0;" src="https://camo.githubusercontent.com/365986a132ccd6a44c23a9169022c0b5c890c387/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6769746875622f726962626f6e732f666f726b6d655f72696768745f7265645f6161303030302e706e67" alt="Fork me on GitHub" data-canonical-src="https://s3.amazonaws.com/github/ribbons/forkme_right_red_aa0000.png">
    </a>
  </embed>


**DeepChem文档项目目的是帮助大家快速掌握DeepChem工具，使大家能够将深度学习技术应用于各自的领域，同时提供一个交流分享平台。**



DeepChem 简介
-----------------

DeepChem 目的是为科学研究领域提高高质量、易用的深度学习工具。
其中工具主要聚焦于化学领域的研究，因此命名为DeepChem。
随着项目的不断发展，其逐渐被应用到其他科学领域的研究。


Github上的 `DeepChem仓库 <https://github.com/deepchem/deepchem>`_ 集成了各种科学工具的套件。随着项目的成熟，越来越多针对具体项目的工具。
DeepChem 主要是用python开发，但我们正在尝试增加对其他语言的支持。 

DeepChem 可以做什么
-------------------
我们可以用DeepChem做很多有趣的事情， 比如：



- 预测类药小分子的溶解度
- 预测小分子与蛋白质靶标的结合亲和力
- 预测简单材料的物理性质
- 分析蛋白质结构并提取有用的描述符
- 计数显微镜图像中的细胞数
- 等等 

注意：DeepChem是一个机器学习的Python库，你可以借助它来解决上述提供的问题。
DeepChem可能没有解决这些问题的现成模型。


随着时间的流逝，我们希望扩大DeepChem的科学应用范围增加更多的模型。 
这意味着我们需要很多帮助！ 如果你是对开源感兴趣的科学家，
请尝试为`DeepChem仓库 <https://github.com/deepchem/deepchem>`_提供更多的建议和代码。


该文档主要包含以下3个部分：1. 快速入门教程；2. 开发者教程； 3. 实践教程。

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 入门教程

   get_started/installation
   get_started/requirements
   get_started/tutorials
   get_started/examples
   get_started/issues

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 开发者教程

   development_guide/licence
   development_guide/scientists
   development_guide/coding
   development_guide/infra

   .. toctree::
   :glob:
   :maxdepth: 1
   :caption: 实践教程

   examples/tutorials/01start.rst 
   examples/tutorials/02start.rst 


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API 接口参考

   api_reference/data
   api_reference/moleculenet
   api_reference/featurizers
   api_reference/splitters
   api_reference/transformers
   api_reference/models
   api_reference/layers
   api_reference/metrics
   api_reference/hyper
   api_reference/metalearning
   api_reference/rl
   api_reference/docking
   api_reference/utils




关于DeepChem作者
------------------
DeepChem的开发人员是由开源贡献者组成。
任何人都可以自由加入并做出贡献！
DeepChem每周都会有开发人员电话会议。 
您可以在我们的 `论坛 <https://forum.deepchem.io/>`_ 上找到 `会议纪要 <https://forum.deepchem.io/search?q=Minutes%20order%3Alatest>`_ 。

DeepChem开发人员会议对公众开放！要收听，请发送电子邮件至bharath.ramsundar@gmail.com进行自我介绍并提出申请。 


.. important::

   | 加入我们的  `gitter社区 <https://gitter.im/deepchem/Lobby>`_  讨论各种DeepChem问题。
   | 注册我们的 `DeepChem论坛 <https://forum.deepchem.io/>`_ 讨论各种科学研究以及开发的问题。 
   

关于文档翻译者
-------------------
热衷于折腾各种技术。
优秀工具的推广和布道者。
目前已翻译并维护的文档有：

1. ` PyMOL中文教程http://pymol.chenzhaoqiang.com/ <http://pymol.chenzhaoqiang.com/>`_
2. ` RDKit中文教程 <http://rdkit.chenzhaoqiang.com/>`_ 
3. ` DeepChem中文教程 <http://deepchem.chenzhaoqiang.com/zh_CN/latest/>`_ 







Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
