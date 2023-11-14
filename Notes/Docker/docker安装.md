- 轻量级的虚拟机环境**Docker**安装的必要性

  - 直接在Linux系统上安装package，package之间会存在冲突，此时可以使用Docker创建出虚拟的空间叫做容器，不同容器之间是相互隔离的。
  - 产品源代码泄露，先编译源代码程序，然后使用编译好的程序上传到目标服务器进行运行，可以避免客户拿到源代码

- 关闭CentOS系统的SELINUX服务

  ```shell
  vi /etc/selinux/config
  
  # 修改为
  SELINUX=disabled
  ```

  

- CentOS 系统安装docker 

  - 离线安装

    [下载地址](https://download.docker.com/linux/static/stable/x86_64/)，https://blog.csdn.net/carefree2005/article/details/130616307

  - 

  ```shell
  # 安装
  yum install docker -y
  
  # 查询当前安装的docker相关包
  rpm -qa | grep docker
  
  # 安装最新版本
  curl -fsSL https://get.docker.com/ | sh
  
  
  sudo systemctl start docker
  
  # 启动
  service docker start
  
  # 重启
  service docker restart
  
  # 关闭
  service docker stop
  
  # 配置Docker加速器
  curl -sSL https://get.daocloud.io/daotools/set_mirror.sh | sh -s http://f1361db2.m.daocloud.io
  
  # 修改文件，去掉逗号
  vi /etc/docker/daemon.json
  
  {"registry-mirrors": ["http://f1361db2.m.daocloud.io"],}
  ```

- 使用报错

  ```shell
  [root@localhost ~]# docker images
  Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
  
  # 重启一下docker服务
  [root@localhost ~]# systemctl daemon-reload
  [root@localhost ~]# systemctl restart docker.service
  ```

  

- docker 使用操作

  先下载镜像（只读），为镜像创建容器，在容器中部署程序。

  - 镜像

    为了快速打包和部署软件环境，Docker引入了镜像机制，镜像是一个配置好的**只读**层软件环境，我们可以使用DockerFile文件创建出镜像，也可以从Docker仓库中下载到镜像。

  - 容器

    容器是在镜像的基础上创建出的虚拟实例，内容**可读可写**，一个Docker镜像可以创建出多个容器，而容器之间相互隔离，部署的程序不会互相干扰，所有的容器直接使用宿主机Linux内核、内存和硬盘，所以容器的性能非常接近于宿主机。

  - Docker常用命令图示

    ![image-20220923203834098](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20220923203834098.png)

    ​	attach 命令为早期使用的命令，目前已经过时；commit命令可以将容器转化为镜像。

- 基本操作

  - 镜像操作

    ```shell
    # 下载镜像
    docker pull python:3.7
    
    # 查看docker环境中安装的镜像
    docker images
    
    # 查看镜像相关的信息
    docker inspect python:3.7
    
    # 导出镜像文件
    docker save python:3.7 > /root/python.tar.gz
    
    
    docker save group_analy_x86:1.0.0 > ./group_analy_x86.tar
    # 删除镜像
    
    docker rmi python:3.7
    
    # 导出镜像文件
    docker save python:3.7 > /root/python.tar.gz
    
    # 导入镜像
    docker load < /root/python.tar.gz
    ```

  

  - 容器操作

    ```shell
    # 初次创建、直接会进入和运行容器
    docker run -it -rm --name=p1 python:3.7 bash
    
    # -it 指定与容器的交互方式
    # --name 指定容器的名称
    # python:3.7 指定使用哪个镜像创建容器，
    # bash 指定容器启动之后执行命令行操作
    
    # 退出容器；创建后使用exit退出容器，会使得容器进行退出状态
    exit
    
    # 查看容器状态
    docker ps -a
    
    # 启动容器
    docker start p1
    
    # 暂停运行容器
    docker pause p1
    
    # 恢复
    docker unpause p1
    
    # 进入运行的容器
    docker exec -it p1 bash
    # 使用上述方式 exit 退出容器，不会使得容器退出
    
    # 查看容器详细信息
    docker inspect p1
    
    # docker 会为每个容器分配一个IPAddress 
    
    
    # 停止容器
    docker stop p1
    
    # 不能直接删除正在运行的容器，可以先停止然后再删除。
    docker rm p1
    
    ```
    
    

- 构建容器技术

  - **网络配置**：docker 环境会自动为容器分配IP地址（默认是动态的），为容器分配固定的IP地址,以便可以进行容器间的相互通信。

    - 可以单独创建一个docker内部的网段（172.18.0.X）

      默认情况下，Docker环境会给容器分配动态的IP地址，这就导致下次启动容器的时候，IP地址就改变了，容器之间的相互调用无法实现了。重新启动容器IP地址会发生改变。
      
      ```shell
      # 创建网段
      docker network create --subnet=172.18.0.0/16 mynet
      
      # 查看docker环境中存在的网段
      docker network ls
      
      # 删除网段（必须先删除网段关联的容器，才能删除掉该网段）
      docker network rm mynet
      
      
      # 创建容器
      docker run -it --name=p1 --net mynet  --ip 172.18.0.2 python:3.7 bash
      
      # 查看容器IP
      ip addr
      
      ```

      进入容器中执行命令 `ip addr` 报错:`bash: ip: command not found` ,可直接执行下面命令进行解决。
      
      ```shell
      apt-get update
      
      apt install -y iproute2
      ```
      
      

  - **端口映射**：将容器的端口映射到宿主机上面，可以通过访问宿主机的端口从而调用docker容器中部署的flask项目。

    默认情况下，除了宿主机之外，任何主机无法远程访问Docker容器。通过端口映射，可以把容器端口映射到宿主机的端口，这样其他主机就能访问容器了，映射到宿主机的端口，不需要设置防火墙规则就可以使用。

    ```shell
    # 其中 5000为容器端口，9500为宿主机端口。
    docker run -it --name=p1 -p 9500:5000 python:3.7 bash
    
    # 可以同时进行多个端口映射
    docker run -it -p 9500:5000 -p 9600:3306 python:3.7 bash
    
    exit
    docker start p1
    docker ps -a
    # 可以发现端口映射生效了。
    
    ```

    

  - **目录挂载**：将python项目放到容器中，可以将宿主机的目录挂载到容器中，此时将文件放在挂载目录中，就可以从容器中看到挂载的文件了，也可以将容器中的文件（如日志文件）放在指定目录，也可以在宿主机上查看该文件内容。

    为了能把一部分业务数据保存在Docker环境之外，或者把宿主机的文件传入容器，所以需要给容器挂载宿主机的目录。

    Docker 环境只支持目录挂载，不支持文件挂载，而且一个容器可以挂载多个目录。

    在创建容器之前，首先要在宿主机上创建一个目录，然后将该目录挂载到python容器上，将来把需要部署的python项目直接存放在目录里边，那么在python容器中就可以看到这个python项目。
    
    ```shell
    mkdir algProjects
    
    # 宿主机目录(前者):/root/algProjects/testProject，容器内的目录可以不存在，会自动创建。 
    docker run -it --name=p1 -v /root/algProjects/testProject:/root/project python:3.7 bash
    
    # 验证
    ls /root/project
    # 可以发现容器内确实存在目录下包含的文件。
    
    cd /root/project
    touch 2.txt
    
    # 退出容器查看宿主机上是否存在2.txt文件
    ls /root/algProjects/testProject
    
    ```
    
    使用以上三种技术运行容器:
    
    ```shell
    docker run -it -d --name=p1 --net mynet  --ip 172.18.0.2 -p 9500:5000 -v /root/algProjects/testProject:/root/project python:3.7 bash
    ```





