### GPU环境配置

- [官方安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

- [多CUDA安装指南](https://www.cnblogs.com/smileglaze/p/16826946.html)

- GPU环境配置三种安装方式：

1. CUDA、pip/conda pytorch，首先安装 CUDA【NVIDIA Driver + CUDA Toolkit + cuDNN】，然后只安装pytorch

2. NVIDIA Driver、conda  pytorch （使用conda 安装cuda toolkit以及cudnn，不是系统级别安装，只是虚拟环境级别安装，安装的并非完整的包）

   ```shell
   nvidia-cublas-cu11       11.10.3.66
   nvidia-cuda-nvrtc-cu11   11.7.99
   nvidia-cuda-runtime-cu11 11.7.99
   nvidia-cudnn-cu11        8.5.0.96
   ```

3. NVIDIA Driver、nvidia  docker 

   ```shell
   # 查找可安装驱动版本
   sudo apt search nvidia-driver | grep nvidia-driver
   # Ubuntu 驱动安装命令
   sudo apt install nvidia-driver
   ```

#### 1.安装驱动Driver

##### 1.1 查看显卡型号

查看显卡型号

```shell
# 查看显卡型号：
nvidia-detect -v

# 查看显卡型号：推荐使用
lspci | grep -i nvidia

# 输出
0000:3b:00.0 3D Controller NVIDIA Corporation Device leb8（rev a1）
0000:37:00.0 3D Controller NVIDIA Corporation Device le78（rev a1）
```

解码获取显卡型号

```shell
# 在 http://pci-ids.ucw.cz/mods/PC/10de?action=help?help=pci 可以查询到 1eb8和le78 对应的显卡版本为 
1eb8	TU104GL [Tesla T4]
1e78	TU102GL [Quadro RTX 6000/8000]
        Subsystems
        Id          Name
        10de 13d8	Quadro RTX 8000	
        10de 13d9	Quadro RTX 6000
```

至于1e78下面是Quadro RTX 6000 还是 Quadro RTX 8000

```shell
lspci -v -s 0000:37:00.0

# 输出：
Subsystem：NVIDIA Corporation Device l3d9
# 根据子系统id可知，显卡型号为Quadro RTX 6000
```

下载、安装驱动

```shell
# 在驱动程序下载 https://www.nvidia.cn/Download/index.aspx?lang=cn 中根据得到显卡型号选择下载驱动程序，得到如下文件：
NVIDIA-Linux-x86_64-410.129-diagnostic.run

# 一般就可以正常执行了。
wget https://cn.download.nvidia.cn/XFree86/Linux-x86_64/525.116.04/NVIDIA-Linux-x86_64-525.116.04.run


```

驱动安装完成后，可以如下命令检验显卡版本

```shell
nvidia-smi -L 

# 输出
GPU 0:Tesla T4 (UUID:GPU-……)
```

##### 1.2 安装kernel相关依赖

```shell
# 查看内核版本
uname -a
# 或
uname -r


# 输出
Linux 内核版本 3.10.0-514.21.2.el7.x86_64


# 下载并安装内核文件，要求两者的版本必须严格一致。
kernel-devel-3.10.0-514.21.2.el7.x86_64.rpm

# 如果无法直接下载到该版本的内核文件，可以在其他连网的Linux系统中使用下面的命令进行下载：
wget https://buildlogs.centos.org/c7.1611.u/kernel/20170620132051/3.10.0-514.21.2.el7.x86_64/kernel-devel-3.10.0-514.21.2.el7.x86_64.rpm --no-check-certificate



.3.10.0-862.el7.x86_64

wget http://ftp.scientificlinux.org/linux/scientific/7.4/x86_64/updates/security/kernel-devel-3.10.0-862.el7.x86_64.rpm 


# 本地离线安装内核
sudo yum install kernel-devel-3.10.0-514.21.2.el7.x86_64.rpm

# 在线连网安装
sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r)

```

##### 1.3 安装驱动

```shell
# 示例1
sh ./NVIDIA-Linux-x86_64-410.129-diagnostic.run --kernel-source-path=/usr/local/kernels/3.10.0-514.21.2.el7.x86_64

# 示例2
sh ./NVIDIA-Linux-x86_64-525.116.04.run --kernel-source-path=/usr/src/kernels/3.10.0-862.el7.x86_64


kernel-devel-3.10.0-862.el7.x86_64


sh ./NVIDIA-Linux-x86_64-535.54.03.run --kernel-source-path=/usr/src/kernels/3.10.0-862.el7.x86_64

# 驱动是否安装成功测试，只安装完Driver，这个命令可以正常使用，而`nvcc -V`则无法正常使用
nvidia-smi

# 如果是正常显示驱动版本信息,则表示安装成功。
```

##### 1.4 卸载驱动

- 卸载方式一：

  ```shell
  sh NVIDIA-Linux-x86_64-418.126.02.run --uninstall
   
  # 重启更新配置信息
  reboot
  ```

- [卸载方式二](https://devicetests.com/uninstall-nvidia-drivers-ubuntu)：

  If you installed the Nvidia driver directly from the Nvidia website, you can use the nvidia-uninstall command to uninstall it. [官方指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

  ```shell
   sudo /usr/bin/nvidia-uninstall #   sudo nvidia-uninstall
  
  # 官方命令 
  # Use the following command to uninstall a Driver runfile installation：
  # sudo /usr/bin/nvidia-uninstall
  
  # 重启更新配置信息
  reboot
  ```

  



#### 2. 安装CUDA Toolkit

##### 2.1 下载与安装

```shell
# 选择CUDA版本：https://developer.nvidia.com/cuda-toolkit-archive    选择runfile格式安装，最方便

wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
sudo sh cuda_11.7.1_515.65.01_linux.run

```

可以看到，本地安装有 runfile 和 rpm 两个选项，两者安装后的效果是一样的，只是在安装时使用的命令不同。

Runfile方式安装时不会进行依赖安装包验证，而 RPM 方式安装时则会进行安装包验证，如果没有满足要求的依赖安装包，则会试图进行安装，但是往往会安装最新版本的依赖安装包，而最新版本往往会和系统不适配，因此最好是先手动安装。

##### 2.2 卸载

```shell
# Use the following command to uninstall a Toolkit runfile installation:

sudo /usr/local/cuda-X.Y/bin/cuda-uninstaller
```



##### 2.3 配置与检测

```shell
sudo /usr/local/cuda-11.8/bin/cuda-uninstaller
vim ~/.bashrc

# 追加
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH


# 保存退出
:wq

# 更新配置
source ~/.bashrc

# 测试
nvcc -V
```

#### 3 安装cuDNN

```shell
# 下载指定版本的cudnn：https://developer.nvidia.com/cudnn

# 下载
https://developer.nvidia.com/compute/cudnn/secure/8.5.0/local_installers/11.7/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz

tar -xvf cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz

# 解压后并进入，拷贝到cuda-11.7，并给所有用户添加读的权限
cd cuda
sudo cp lib64/* /usr/local/cuda-11.7/lib64/
sudo cp include/* /usr/local/cuda-11.7/include/
sudo chmod a+r /usr/local/cuda-11.7/lib64/*
sudo chmod a+r /usr/local/cuda-11.7/include/*



cd /usr/local/cuda-11.6/lib64/
sudo rm -rf libcudnn.so libcudnn.so.8
sudo ln -s libcudnn.so.8.4.1 libcudnn.so.8
sudo ln -s libcudnn.so.8 libcudnn.so
sudo ldconfig -v


```

查看cuDNN版本

```shell
cat /usr/local/cuda-11.7/include/cudnn.h | grep CUDNN_MAJOR -A2
```

- 安装conda

  ```shell
  wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh
  ```

- 传统上，安装 NVIDIA Driver 和 CUDA Toolkit 的步骤是分开的，但实际上我们可以直接安装 CUDA Toolkit，系统将自动安装与其版本匹配的 NVIDIA Driver。反之，如果先安装Driver，再安装 Tookit 时系统会报错。因此正确的安装方式就是直接安装 CUDA Toolkit，正确安装之后。

  