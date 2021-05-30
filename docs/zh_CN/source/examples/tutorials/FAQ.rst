FAQ 
======================================
这里收集了在使用DeepChem过程中遇到的一些常见问题。

常见问题
----------------

.. contents:: 目录
    :local:


numpy 版本不兼容问题
--------------------
报错：

.. code-block:: console

    NotImplementedError: Cannot convert a symbolic Tensor (gradient_tape/private__graph_conv_keras_model_1/graph_gather_1/sub:0) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported


解决办法：



numpy版本过高,conda安装指定版本的numpy,1.19.5即可解决问题。

.. code-block:: console

    conda install numpy=1.19.5 -c conda-forge




