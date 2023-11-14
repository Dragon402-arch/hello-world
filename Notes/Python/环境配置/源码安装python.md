## Python源码安装

###  1. CPU架构

- CPU架构：
  - ARM：鲲鹏计算（华为云 / 弹性云服务器），精简指令集，aarch64表示ARM
  - X86（主流），全套指令集

### 2. Python安装

#### 2.1 下载Python安装包

```shell

# Miniconda 包下载地址
https://docs.conda.io/en/latest/miniconda.html#linux-installers

# Python包下载地址
https://www.python.org/downloads/source/

# 下载得到
Python-3.7.10.tgz

# 上传到服务器并解压
tar -xzvf ./Python-3.7.10.tgz -C /root/

```

#### 2.2 联网环境下安装依赖包

```shell
# 在CentOS 系统中准备编译环境
yum -y install zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel libffi-devel gcc make

yum install libxml2-devel xmlsec1-devel xmlsec1-openssl-devel libtool-ltdl-devel

yum install lapack lapack-devel blas blas-devel
```

#### 2.3 配置并安装

##### 2.3.1 **默认配置安装**

```shell
# 进入到解压后的Python-3.7.10目录下，执行如下命令
cd Python-3.7.10

# 默认安装在/usr/local路径下面
./configure


make && make install

# 至此安装完毕
```

- 默认配置安装路径

  ```shell
  --prefix=PREFIX install architecture-independent files in PREFIX[/usr/1ocal]
  
  
  By default,'make install' will install all the files in','/usr/local/bin','/usr/local/lib’ etc.You can specify installation prefix other than'/usr/local' using'--prefixor',For instance 'instance--prefix=$HOME'.
  ```

##### 2.3.2 **自定义配置安装**

```shell
# 进入到解压后的Python-3.7.10目录下，执行如下命令

./configure --prefix=/root/Python-3.7.10

# --prefix是Python的安装目录。

make && make install

# 至此安装完毕

```

- 报错

  ```shell
  # no acceptable C compiler found in $PATH
  # 解决方案：缺少gcc系统包
  # https://zhuanlan.zhihu.com/p/496126731
  
  
  zlib
  
  # ModuleNotFoundError:no module named '_ssl'
  # 解决方案：https://blog.csdn.net/qq_42805358/article/details/103991134
  # 编译的时候加上ssl路径
  ./configure --prefix=/usr/local/python3.7 --enable-optimizations --with-openssl=/usr/lib/ssl
  
  ```
  
  

#### 2.4 创建软链接

##### 2.4.1 默认配置场景

- Python3 相关

  ```shell
  # 软连接的路径
  cd /usr/bin
  
  # 查看python相关的软连接
  ls -al *python*
  
  # 删除软链接
  sudo rm -rf  /usr/bin/python3
  
  # 添加软链接
  sudo ln -s /usr/local/bin/python3  /usr/bin/python3
  ```

  - 修改环境变量

    ```shell
    # 默认安装位置 /usr/local/
     
    vim ~/.bash_profile
    
    
    export PYTHON_HOME=/usr/local # python 安装目录
    export PATH=$PYTHON_HOME/bin:$PATH
    
    # 使配置生效
    source ~/.bash_profile 
    
    ```

  - 测试：`python3`，查看输出Python版本

- pip 相关

  ```shell
  # 查看pip相关的软连接
  cd /usr/bin
  
  ls -al *pip*
  
  # 删除软连接
  sudo rm -rf  /usr/bin/pip3
  
  # 添加软连接
  sudo ln -s /usr/local/lib/python3.7/site-packages/pip /usr/bin/pip3
  
  ```

  - 修改 pip 镜像源

    ```sh
    pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/
    pip3 config set install.trusted-host mirrors.aliyun.com
    ```

    - 阿里云：https://mirrors.aliyun.com/pypi/simple/
    - 中国科技大学：https://pypi.mirrors.ustc.edu.cn/simple/
    - 豆瓣：http://pypi.douban.com/simple/
    - 清华大学：https://pypi.tuna.tsinghua.edu.cn/simple/
    - 中国科学技术大学： http://pypi.mirrors.ustc.edu.cn/simple/

##### 2.4.2 自定义配置场景

- python3相关

  ```shell
  # 软连接的路径
  cd /usr/bin
  
  # 查看python相关的软连接
  ls -al *python*
  
  # 删除软链接
  sudo rm -rf  /usr/bin/python3
  
  # 添加软连接
  sudo ln -s /root/Python-3.7.10/bin/python3  /usr/bin/python3
  ```

  - 修改环境变量

    ```shell
     vim ~/.bash_profile
    
    
    export PYTHON_HOME=/root/Python-3.7.10 # python 安装目录
    export PATH=$PYTHON_HOME/bin:$PATH
    
    # 使配置生效
    source ~/.bash_profile
    ```

  - 测试：`python3`，查看输出Python版本

- pip 相关

  ```sh
  # 查看pip相关的软连接
  cd /usr/bin
  
  ls -al *pip*
  
  # 删除软连接
  sudo rm -rf  /usr/bin/pip3
  
  # 添加软连接
  sudo ln -s /root/Python-3.7.10/lib/python3.7/site-packages/pip /usr/bin/pip3
  ```

  - 修改pip镜像源

    ```shell
    pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/
    pip3 config set install.trusted-host mirrors.aliyun.com
    ```

  - 工具函数

    ```shell
    # 工具函数：查找某个文件所在的路径
    find / -name 'pip.py'
    find / -name 'pip' -type d # 查找文件夹或目录
    whereis pip
    ```

### 3. Python虚拟环境创建

#### 3.1 虚拟环境文件夹目录结构

```shell
# 将虚拟环境下python3.7整个文件夹替换掉
└──nlp
    ├──bin
    ├──include
    ├──lib
    │   └──python3.7 (整体替换掉)
    │         └──site-packages(虚拟环境下安装包存放目录)
    ├──lib64
    └──pyvenv.cfg
```

#### 3.2 在来源服务器上创建虚拟环境

```shell
mkdir algProjects

cd algProjects

# 创建环境
python3 -m venv nlp

# 激活环境
source ./nlp/bin/activate 

# 安装所有依赖包，一个package安装失败，全部安装失败
pip install -r requirements.txt

# 将虚拟环境下安装的packages打包
tar -zcvf ./nlp.tar.gz ./nlp
```

#### 3.3 在目标服务器上创建并复制虚拟环境

##### 3.3.1 创建虚拟环境

```shell
mkdir algProjects

cd algProjects

# 创建环境
python3 -m venv nlp
```

##### 3.3.2 复制虚拟环境

```shell
# 删除文件
cd ./nlp/lib/python3.7

rm -rf *

cd ..

rmdir python3.7

# 解压文件
tar -zxvf ./nlp.tar.gz -C .

# 完成依赖包替换
cp -r ./nlp/lib/python3.6  .

# 删除当前目录下的nlp文件夹

cd ./nlp

rm -rf *

cd ..

rmdir nlp




# 项目操作
mkdir algProjects

cd algProjects

# 创建环境
python3 -m venv search_venv

# 删除文件
cd ./search_venv/lib/python3.7

rm -rf *

cd ..

rmdir python3.7

# 解压文件到./search_venv/lib/python3.7到同级目录下
tar -zxvf ./search_venv_11.tar.gz -C .

# 完成依赖包替换
cp -r ./search_venv_11/lib/python3.7  .

# 删除当前目录下的search_venv_11文件夹

cd ./search_venv_11

rm -rf *

cd ..

rmdir search_venv_11


```

- 测试

  ```shell
  # 激活环境
  source ./nlp/bin/activate 
  
  # 测试替换是否成功
  pip list
  ```

  

![image-20220420163052411](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20220420163052411.png)

