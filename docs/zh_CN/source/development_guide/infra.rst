基础架构 （Infrastructures）
===============================

DeepChem项目的基础架构分布在不同服务上，
比如github\Conda Forge\Docker Hub\PyPI\Amazon Web Services等。 
该基础结构由DeepChem开发团队维护。 

GitHub
---------

DeepChem的核心代码在`deepchem <https://github.com/deepchem>`_ GitHub组织中维护。
而且，我们使用GitHub Actions建立了一个**持续的集成管道**。 


DeepChem开发人员具有对该存储库中文件具有写访问权限，技术指导委员会成员具有管理访问权限。 



Conda Forge
--------------

DeepChem的仓库`feedstock <https://github.com/conda-forge/deepchem-feedstock>`_ 中维护了
构建conda-forge的deepchem包的方法。



Docker Hub
------------
DeepChem 在 `Docker Hub <https://hub.docker.com/r/deepchemio/deepchem>`_托管了主要版本和最新版本的镜像文件.



PyPI
-------
DeepChem也在`PyPI <https://pypi.org/project/deepchem/>`_提供了各种不同版本。

亚马逊web 服务（AWS）
----------------------
DeepChem的网站基础设施都是通过不同的AWS 服务器管理的。
所有DeepChem开发人员都可以通过deepchem-developers IAM角色访问这些服务。（IAM角色控制访问权限。） 

目前，@ rbharath是唯一具有IAM角色管理员访问权限的开发人员，
但从长远来看，我们应该迁移此角色，以便其他人可以使用这些角色。 


亚马逊的S3服务器
^^^^^^^^^^^^^^^^
亚马逊的S3允许在“buckets”上存储数据
（可以把buckets当成是文件夹）。
Deepchem S3的buckets有两个核心数据： 

  - deepchemdata: 该buckets包含MoleculeNet数据集，预特征化的数据集和预先训练的模型。 

  - deepchemforum:该buckets托管论坛的备份。
   出于安全原因，该buckets是私有的。
     论坛本身托管在Digital Ocean云服务器上，目前只有仅@rbharath可以访问的。
     从长远来看，我们应该将论坛迁移到AWS上，以便所有DeepChem开发人员都可以访问该论坛。
     The forums themselves are a discord instance. 
      论坛每天一次将其备份上传到此S3的buckets中。
     如果论坛崩溃，则可以从此buckets中的备份中还原它们。 
 

AWS的Route 53云域名系统
^^^^^^^^^^^^^^^^^^^^^^
deepchem.io网站的DNS由Route 53处理。
“托管区域” deepchem.io保存了该网站的所有DNS信息。 

AWS的证书管理
^^^^^^^^^^^^^^^^^^^

AWS证书管理器为*.deepchem.io和 deepchem.io 域名提供SSL / TLS证书。

GitHub Pages
^^^^^^^^^^^^^^^
我们利用GitHub Pages服务部署我们的静态网站。

GitHub Pages连接到AWS证书管理器中的证书。
我们为www.deepchem.io设置了CNAME，为deepchem.io设置了A记录。 
DeepChem 的GitHub Pages 地址是  [deepchem/deepchem.github.io](https://github.com/deepchem/deepchem.github.io).

GoDaddy域名供应商
--------------------
deepchem.io域名在GoDaddy域名供应商那边注册的。
如果您在AWS Route 53中更改名称服务器，您将需要更新GoDaddy记录。
目前，只有@rbharath可以访问拥有deepchem.io域名的GoDaddy帐户。
我们应该探索如何为其他DeepChem开发人员提供对域名的访问权限。 

Digital Ocean云主机提供商
---------------------------
DeepChem的论坛在Digital Ocean云主机提供商的实例机器上托管。
目前，只有@rbharath可以访问此实例。
我们应该将此实例迁移到AWS上，以便其他DeepChem开发人员可以帮助维护论坛。 



