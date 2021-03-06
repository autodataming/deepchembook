已知问题和局限性
--------------------------

损坏的功能
^^^^^^^^^^^^^^^

DeepChem中有少量的功能已经是损坏的，不能使用了。
DeepChem团队会对其中的功能进行修复或者移除。
在一个大的项目中，比如DeepChem, 找出所有存在的bug几乎不可能，
但我们希望通过列出我们知道部分或全部损坏的功能来为减少你在使用过程中遇到的麻烦。



*注意:此列表可能并不详尽。 如果我们错过了一些问题，
请在 [github issue2376](https://github.com/deepchem/deepchem/issues/2376)告诉我们。*

+--------------------------------+-------------------+---------------------------------------------------+
| Feature                        | Deepchem response | Tracker and notes                                 |
|                                |                   |                                                   |
+================================+===================+===================================================+
| ANIFeaturizer/ANIModel         | Low Priority      | The Deepchem team recommends using TorchANI       |
|                                | Likely deprecate  | instead.                                          |
|                                |                   |                                                   |
+--------------------------------+-------------------+---------------------------------------------------+

实验性的功能
^^^^^^^^^^^^^^^^^^^^^
Deepchem功能通常会经过严格的代码审查和测试，以确保他们可以用于生产环境。 

以下Deepchem功能尚未在其他Deepchem模块的水平上进行全面测试，在生产环境中使用可能存在问题。 



*注意:此列表可能并不详尽。 如果我们错过了一些问题，
请在 [github issue2376](https://github.com/deepchem/deepchem/issues/2376)告诉我们。*

+--------------------------------+---------------------------------------------------+
| Feature                        | Tracker and notes                                 |
|                                |                                                   |
+================================+===================================================+
| Mol2 Loading                   | Needs more testing.                               |
|                                |                                                   |
|                                |                                                   |
+--------------------------------+---------------------------------------------------+
| Interaction Fingerprints       | Needs more testing.                               |
|                                |                                                   |
|                                |                                                   |
+--------------------------------+---------------------------------------------------+

如果您想帮助我们解决这些已知问题，请考虑为成为Deepchem的贡献者！ 

