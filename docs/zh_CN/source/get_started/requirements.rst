依赖
------------

必须依赖模块
^^^^^^^^^^^^^^^^^

DeepChem支持python 3.6和python 3.7。
运行DeepChem必须的依赖模块有：


- `joblib < https://pypi.python.org/pypi/joblib>`_ :对于大数据而言，joblib比pickle更加高效，但是joblib只能将对象存储在磁盘文件中，不能保存为字符串。
- `NumPy <https://numpy.org/>`_
- `pandas <http://pandas.pydata.org/>`_
- `scikit-learn <https://scikit-learn.org/stable/>`_
- `SciPy <https://www.scipy.org/>`_
- `TensorFlow <https://www.tensorflow.org/>`_

  - `deepchem>=2.4.0` 需要 TensorFlow v2 (2.3.x) 推荐
  - `deepchem<2.4.0`  需要 TensorFlow v1 (>=1.14)


可选依赖项
^^^^^^^^^^^^^^^^^

DeepChem 有大量的可选依赖模块包：


+--------------------------------+---------------+---------------------------------------------------+
| 包名字                | 包的版本       | DeepChem 哪个文件使用依赖模块              |
|                                |               | (dc: deepchem)                                    |
+================================+===============+===================================================+
| `BioPython <https://biopython.org/wiki/Documentation>`_                   | latest        | :code:`dc.utlis.genomics_utils`                   |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `Deep Graph Library <https://www.dgl.ai/>`_          | 0.5.x         | :code:`dc.feat.graph_data`,                       |
|                                |               | :code:`dc.models.torch_models`                    |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `DGL-LifeSci <https://github.com/awslabs/dgl-lifesci>`_                 | 0.2.x         | :code:`dc.models.torch_models`                    |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `HuggingFace Transformers <https://huggingface.co/transformers/>`_    | Not Testing   | :code:`dc.feat.smiles_tokenizer`                  |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `LightGBM <https://lightgbm.readthedocs.io/en/latest/index.html>`_                    | latest        | :code:`dc.models.gbdt_models`                     |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `matminer <https://hackingmaterials.lbl.gov/matminer/>`_                    | latest        | :code:`dc.feat.materials_featurizers`             |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `MDTraj <http://mdtraj.org/>`_                      | latest        | :code:`dc.utils.pdbqt_utils`                      |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `Mol2vec <https://github.com/samoturk/mol2vec>`_                     | latest        | :code:`dc.utils.molecule_featurizers`             |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `Mordred <http://mordred-descriptor.github.io/documentation/master/>`_                     | latest        | :code:`dc.utils.molecule_featurizers`             |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `NetworkX <https://networkx.github.io/documentation/stable/index.html>`_                    | latest        | :code:`dc.utils.rdkit_utils`                      |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `OpenAI Gym <https://gym.openai.com/>`_                  | Not Testing   | :code:`dc.rl`                                     |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `OpenMM <http://openmm.org/>`_                      | latest        | :code:`dc.utils.rdkit_utils`                      |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `PDBFixer <https://github.com/pandegroup/pdbfixer>`_                    | latest        | :code:`dc.utils.rdkit_utils`                      |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `Pillow <https://pypi.org/project/Pillow/>`_                      | latest        | :code:`dc.data.data_loader`,                      |
|                                |               | :code:`dc.trans.transformers`                     |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `PubChemPy <https://pubchempy.readthedocs.io/en/latest/>`_                   | latest        | :code:`dc.feat.molecule_featurizers`              |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `pyGPGO <https://pygpgo.readthedocs.io/en/latest/>`_                      | latest        | :code:`dc.hyper.gaussian_process`                 |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `Pymatgen <https://pymatgen.org/>`_                    | latest        | :code:`dc.feat.materials_featurizers`             |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `PyTorch <https://pytorch.org/>`_                     | 1.6.0         | :code:`dc.data.datasets`                          |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_           | 1.6.x (with   | :code:`dc.feat.graph_data`                        |
|                                | PyTorch 1.6.0)| :code:`dc.models.torch_models`                    |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `RDKit <http://www.rdkit.org/docs/Install.html>`_                       | latest        | Many modules                                      |
|                                |               | (we recommend you to instal)                      |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `simdna <https://github.com/kundajelab/simdna>`_                      | latest        | :code:`dc.metrics.genomic_metrics`,               |
|                                |               | :code:`dc.molnet.dnasim`                          |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `Tensorflow Probability <https://www.tensorflow.org/probability>`_      | 0.11.x        | :code:`dc.rl`                                     |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `Weights & Biases <https://docs.wandb.com/>`_            | Not Testing   | :code:`dc.models.keras_model`,                    |
|                                |               | :code:`dc.models.callbacks`                       |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `XGBoost <https://xgboost.readthedocs.io/en/latest/>`_                     | latest        | :code:`dc.models.gbdt_models`                     |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `Tensorflow Addons <https://www.tensorflow.org/addons/overview>`_           | latest        | :code:`dc.models.optimizers`                      |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
          









