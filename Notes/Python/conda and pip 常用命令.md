#### Conda 常用命令

- 创建环境

  ```shell
  conda acreate -n pytest python=3.8
  ```

  另外的创建方式，使用 `-p`  时不能再使用 `-n`，

  ```shell
  conda create -p=/home/lis/miniconda3/envs/test python=3.8
  conda create --prefix=/home/lis/miniconda3/envs/test python=3.8
  #  -p PATH, --prefix PATH Full path to environment location (i.e. prefix)
  ```

- 创建虚拟环境时指定环境存放路径

  ```shell
  # 指定增加虚拟环境存放路径
  conda config --add envs_dirs D:\python36-venv
  
  conda config --add envs_dirs /date/lis/conda_envs/
  ```

- 克隆环境 

  ```shell
  conda create -n  hotwords --clone finetune
  ```

  克隆 finetune 虚拟环境，并由此创建 hotwords 环境

- 修改环境名称

  ```shell
  conda rename -n old_name new_name
  ```

- 删除环境 

  ```shell
  conda remove --name  finetune --all 
  ```

- 添加虚拟环境存放路径

  ```shell
  conda config --add envs_dirs 
  ```

- 复制过来的虚拟环境 下使用 clear 命令报错：terminals database is inaccessible，解决办法：

  ```shell
  conda update ncurses
  ```

  

#### pip 常用命令

- 导出环境包

  ```shell
  pip freeze > requirements.txt 
  ```

- 安装环境包

  ```shell
  pip install -r requirements.txt 
  ```

- 升级某个环境包到最新版本

  ```shell
   python -m pip install --upgrade transformers
   
   pip install --upgrade transformers
  ```

- 修改镜像源

  ```shell
  pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
  pip config set install.trusted-host mirrors.aliyun.com
  ```

  方式二：

  在Anaconda或Miniconda的安装目录下会存在一个 `.config` 文件夹（使用 `ll -a` 命令可以看到），修改 `./config/pip/pip.conf` 即可配置镜像源。

  ```shell
  [global]
  index-url = https://mirrors.aliyun.com/pypi/simple/
  
  [install]
  trusted-host = mirrors.aliyun.com
  
  ```

  

- 创建虚拟环境

  ```shell
  # 创建虚拟环境
  python -m venv test_venv
  
  # 激活虚拟环境
  source activate /algProject/test_venv/bin
  ```
