#### 1、安装 nvidia-container-toolkit

需要先安装 NVIDIA 驱动

安装指南：https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

```shell
# docker 测试
$ sudo docker run --rm hello-world


# Setting up NVIDIA Container Toolkit

$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo


$ sudo yum clean expire-cache
$ sudo yum install -y nvidia-container-toolkit
$ sudo nvidia-ctk runtime configure --runtime=docker
$ sudo systemctl restart docker

# nvidia-container-toolkit 安装结果测试
$ sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi

docker run --rm --runtime=nvidia --gpus all pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime nvidia-smi

```



#### 2、下载模型权重文件

```shell

# 下载两个模型相关文件放在/date/pretrained_models/kbqa_models/目录下：
	text2vec-large-chinese
	chatglm2-6b
```

下载chatglm2模型文件

```python
# pip install modelscope


def download_chatglm2():
    from modelscope.hub.snapshot_download import snapshot_download

    model_dir = snapshot_download(
        'ZhipuAI/chatglm2-6b',
        cache_dir='/date/pretrained_models/kbqa_models/chatglm2-6b',
        revision="v1.0.2")


def test_chatglm2():
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    from transformers import AutoTokenizer, AutoModel
    pretrained_model_path = "/date/pretrained_models/kbqa_models/chatglm2-6b"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(pretrained_model_path, trust_remote_code=True).half().cuda()
    model = model.eval()
    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)
    response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
    print(response)
    
    

def download_text2vec():
    from modelscope.hub.snapshot_download import snapshot_download

    model_dir = snapshot_download(
        'thomas/text2vec-large-chinese',
        cache_dir='/date/pretrained_models/kbqa_models/text2vec-large-chinese',
        revision="v1.0.0")
```

#### 3、安装Milvus向量数据库

```shell
wget https://github.com/milvus-io/milvus/releases/download/v2.1.4/milvus-standalone-docker-compose.yml -O docker-compose.yml


# In the same directory as the docker-compose.yml file, start up Milvus by running:
# 启动
sudo docker-compose up -d

# 查看状态
sudo docker-compose ps

# 退出
sudo docker-compose down

```

#### 4、下载和构建镜像

修改kbqa_app中kbqa_config中milvus的配置信息：host 和 port

pytorch镜像下载地址：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags

```shell
docker pull pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
```

pytorch版本以及CUDA版本选择：https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-06.html#rel-23-06

```shell
# 下载基础镜像
sudo docker run --gpus all -it --rm -p 8888:8888 -v /home/lis/algProjects:/home/lis/algProjects nvcr.io/nvidia/pytorch:22.07-py3

# 退出容器
exit

# 在Dockerfile所在目录下执行：创建kbqa镜像
docker build -t kbqa:1.0.0 .

/date/pretrained_models/chatglm2_int4_model/

```

注意：操作容器安装一些 python package后，如果没有导出保存镜像，再次运行容器时新安装的包不会保存下来。

#### 5、运行容器

修改代码配置

```shell
# 测试：交互式运行容器kbqa_app
docker run --gpus all -it --name=kbqa_app --rm -p 6787:6787 -v /date/pretrained_models/kbqa_models/:/date/pretrained_models/kbqa_models/ --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 kbqa:1.0.0
  
# 后台执行

docker run --gpus all -d --name=kbqa_app -p 6787:6787 -v /date/pretrained_models/kbqa_models/:/date/pretrained_models/kbqa_models/ --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 kbqa:1.0.0
```



#### 6、压缩镜像

Alpaca Dockerfile 制作 ：https://github.com/tloen/alpaca-lora

```dockerfile
FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04

# 指定时区 
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime &&  echo "Asia/Shanghai" > /etc/timezone

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    curl \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.8-dev  python3.8-distutils \
    && apt install -y build-essential  \ # 安装gcc，编译安装部分python包时需要
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt requirements.txt

RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
	&& python3.8 get-pip.py  --force-reinstall \
    && python3.8 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    
COPY ./kbqa_app ./kbqa_app

ENTRYPOINT [ "python3.8"]
```

创建镜像：

```shell
docker build -t torch1.13.1-cuda11.6-cudnn8-runtime-kbqa:1.0.0 .
docker build -t torch1.13.1-cuda11.6-cudnn8-runtime-triplet:1.0.0 .
```

测试镜像：

```shell
docker run  --gpus=all --rm  -it --entrypoint=bash torch1.13.1-cuda11.6-cudnn8-runtime-kbqa:1.0.0 

docker run --gpus '"device=0,1"' --rm  -it --entrypoint=bash torch1.13.1-cuda11.6-cudnn8-runtime-triplet:1.0.0 

docker run --gpus '"device=0,1"' --rm  -it --entrypoint=bash torch1.13.1-cuda11.6-cudnn8-runtime-kbqa:1.0.0 
```

导入导出镜像：

```shell

# 导出镜像
docker save torch1.13.1-cuda11.6-cudnn8-runtime-kbqa:1.0.0 > ./torch1.13.1-cuda11.6-cudnn8-runtime-kbqa.tar 

# 加载镜像
docker load < ./torch1.13.1-cuda11.6-cudnn8-runtime-kbqa.tar
```

运行容器：

```shell
# 创建并运行容器
docker run --gpus all -d --shm-size 64g --name=kbqa_app -p 6787:6787 -v /date/pretrained_models/kbqa_models/:/date/pretrained_models/kbqa_models/ --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 torch1.13.1-cuda11.6-cudnn8-runtime-kbqa:1.0.0  ./kbqa_app/kbqa_main.py --port 6787 




docker run --gpus all -d --shm-size 64g --name=kbqa_apppp -p 6786:6786 -v /date/pretrained_models/kbqa_models/:/date/pretrained_models/kbqa_models/ --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 torch1.13.1-cuda11.6-cudnn8-runtime-kbqa:1.0.0  ./kbqa_app/kbqa_main.py --port 6786 



docker run --gpus '"device=1"' --rm  -it --shm-size 64g --name=triplet_apppp -p 6878:6878 -v /date/pretrained_models/Qwen-7B-Chat:/date/pretrained_models/Qwen-7B-Chat --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 torch1.13.1-cuda11.6-cudnn8-runtime-triplet:1.0.0  ./triplet_app/router.py


```



