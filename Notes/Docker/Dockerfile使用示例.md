-   router.py文件内容

  ```shell
  #  -*- coding: utf-8 -*-
  
  from flask import Flask, request
  
  app = Flask(__name__)
  
  
  @app.route("/test", methods=["GET"])
  def hello(): 
      name = request.args.get('name')
      return f"{name},love you！"
  
  
  if __name__ == "__main__":
      app.run(host="0.0.0.0", port=5000,debug=True)
  ```

  

- Dockerfile文件

  ```dockerfile
  FROM python:3.7-slim
  
  # 切换到镜像中的指定路径，设为工作目录
  WORKDIR /root/app
   
  COPY requirements.txt .
  
  RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  
  ADD ./app /root/app
  
  
  # Run router.py when the container launches
  CMD ["python", "/root/app/router.py"]
  ENTRYPOINT ["python"]
  CMD ["/apps/compose_test/router.py"]
  # CMD ["python", "router.py"]
  # CMD python /root/app/router.py 这种方式不支持传参
  
  
  
  
  ```

  - 执行命令：
    - RUN 是在 docker build 时运行。
    
    - CMD 在docker run 时运行。
      - 为启动的容器指定默认要运行的程序，程序运行结束，容器也就结束。**CMD 指令指定的程序可被 docker run 命令行参数中指定要运行的程序所覆盖。**如用 `python /root/app/main.py` 进行替代容器运行时默认的命令。
      
    - ENTRYPOINT在docker run 时运行。
      - 类似于 CMD 指令，但其不会被 docker run 的命令行参数指定的指令所覆盖，而且这些命令行参数会被当作参数送给 ENTRYPOINT 指令指定的程序。
      
    
    也就是使用ENTRYPOINT来写死固定的命令，除非你使用 docker run --entrypoint=ls apitest:1.0.0 /opt/app 来修改固定的命令。否则你就只能修改参数，执行固定好的python命令，从 router.py 切换到 main.py可以，而ls  /opt/app 则是不允许的。
    
  - 添加文件: 
    
    - WORKDIR 中指定的路径为Docker环境镜像中的指定路径，当使用 `docker exec -it api_test bash` 进入容器中时，初次进入所处的目录正是指定的目录。在 WORKDIR 中需要使用绝对路径，如果镜像中对应的路径不存在，会自动创建此目录
    - COPY 与 ADD 两个命令功能相似，COPY指令不支持从远程URL获取资源，只能从执行docker build所在的主机上读取资源并复制到镜像中；而ADD指令支持从远程URL获取资源，可以通过URL从远程服务器读取资源并复制到镜像中。

- requirements.txt文件

  ```shell
  click==8.1.3
  Flask==2.2.2
  importlib-metadata==4.12.0
  itsdangerous==2.1.2
  Jinja2==3.1.2
  MarkupSafe==2.1.1
  typing_extensions==4.3.0
  Werkzeug==2.2.2
  zipp==3.8.1
  ```

- 创建镜像

  ```shell
  # -t ：指定要创建的目标镜像名
  docker build -t imagename Dockerfilepath
  
  # 创建镜像，执行该该命令时要确保Dockerfile文件在当前路径下。
  
  docker build -t apitest:1.0.0 .
  # 或
  docker build -t apitest:1.0.0 /root/algProjects/api_test
  
  
  docker build -t graph_algo-1.1.0-py3.9-x86:1.1.0 .
  
  docker build -t event_extract:1.0.0 .
  docker build -t apptest:1.1.0 .
  
  docker run -it --name=p2 test:1.0.0 bash
  
  # 验证是否出现新建的镜像
  docker images         
  
  # 创建并后台运行容器
  docker run -it -d --name=api_test --net mynet  --ip 172.18.0.4 -p 9600:5000 apitest:1.0.0
  
  docker run -d --name=event_extraction_app  -p 7905:7969 event_extraction:1.0.0
  
  
  docker run -d --name=hotwords_app  -p 7301:7301 hotwords:1.0.0
  
  docker run --name=app_test -p 9600:5000 apptest:1.0.0 ls 
  
  
  docker run -d --name=group_analy_app -p 9527:9527 group_analy_x86:1.0.0
  
  
  docker run -d --name=event_extraction_app  -p 7905:7969 event_extraction:1.0.0
  
  
  # 进入容器中
  docker exec -it api_test bash
  
  # 查看文件是否存放在指定位置
  
  
  # 将环境中安装的包写入requirements文件中
  pip freeze > requirements.txt 
  
  
  ```

  

- 运行容器

  ```shell
  docker run -d --name=api_test --net mynet  --ip 172.18.0.4 -p 9600:5000 apitest:1.0.0
  
  ```

- 接口调用

  ```shell
  http://192.168.56.108:9500?text=lee
  ```

- 查看容器日志信息

  ```shell
  docker logs -f --tail 10 a0e3e48c6848 
  
  docker logs -f --tail 10 event_extract
  
  
   docker logs faa0073e66e5
  ```

- 异常报错

  ![image-20230418145112089](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230418145112089.png)

  - 解决方案一：` RUN apt-get update && apt-get -y install gcc`，gcc安装速度极慢
  - 解决方案二：不要使用python:3.7-slim镜像，而是使用python:3.7镜像

- 容器与宿主机之间的文件复制

  ```shell
  # 将宿主机的文件复制到docker环境中,宿主机可以使用相对路径
  
  docker cp /opt/event_template.py event_extraction_app:/root/alg/event_template.py # event_extraction_app 是一个容器名称
   
  
  
  # 将docker环境中的文件复制到宿主机
  docker cp event_extraction_app:/root/alg/event_template.py /opt
  
  ```

  更新项目文件之后，不需要重新创建镜像和容器，只需要重启容器即可。

- 取出镜像中的源码文件夹

  ```shell
  # 查看源码在docker环境中的目录
  docker run --name=app_test -p 9600:5000 apptest:1.0.0 pwd
  
  # 输出 /root/src
  
  # 将项目文件从docker环境中传出
  docker cp app_test:/root/src .
  ```

  表面上掩盖项目文件，实际上还是可以偷到。

  ```shell
  # 此时使用下面的命令就无法查看目录了，就会报错
  docker run --name=app_test1 -p 9600:5000 apptest:1.1.0 pwd
  
  # 可以改为下面的：
  docker run --entrypoint=pwd --name=app_test1 -p 9600:5000 apptest:1.1.0
  ```

  真正运行

  ```shell
  docker run --name=app_test2 -p 9600:5000 apptest:1.1.0
  ```

  

- 更新镜像名称

  ```shell
  docker tag event_extraction:latest event_extraction:1.1.0 
  ```

  

- 创建镜像的考虑：

  - 项目不大且依赖较少，此时可以将项目文件和依赖包都添加到镜像中，然后导出镜像。

  - 项目较大或依赖较多，此时可以将依赖包添加到镜像中，然后将项目文件进行目录挂载，然后运行容器。

    ```shell
    docker run -v /opt/test_app:/root/test_app -w /root/test_app python:3.7 python test.py
    
    
    docker run -d --name=hotwords  -p 7301:7301 -v /home/chinaoly/lis/hotwords_app_src/hotwords_app:/root/algProjects/hotwords_app hotwords_app:1.0.0
    
    
    # -v 进行目录挂载 将宿主机的/opt/test_app目录挂载到容器的/root/test_app目录
    # -w 指定工作目录 指定容器的/root/test_app目录为工作目录。
    # python:3.7 运行容器所用的镜像名称
    # python test.py 运行容器时执行的命令。
    ```
    
    

- 将容器连接到网络

  可以按名称或ID连接容器。 一旦连接，容器可以与同一网络中的其他容器通信。

  ```shell
  # 语法:docker network connect [OPTIONS] NETWORK CONTAINER
  docker network connect mynet app_test
  
  ```

  - 相关命令

    | 命令名称                                                     | 说明                         |
    | ------------------------------------------------------------ | ---------------------------- |
    | [docker network connect](http://www.yiibai.com/docker/network_connect.html) | 将容器连接到网络             |
    | [docker network create](http://www.yiibai.com/docker/network_create.html) | 创建一个网络                 |
    | [docker network disconnect](http://www.yiibai.com/docker/network_disconnect.html) | 断开容器的网络               |
    | [docker network inspect](http://www.yiibai.com/docker/network_inspect.html) | 显示一个或多个网络的详细信息 |
    | [docker network ls](http://www.yiibai.com/docker/network_ls.html) | 列出网络                     |
    | [docker network prune](http://www.yiibai.com/docker/network_prune.html) | 删除所有未使用的网络         |
    | [docker network rm](http://www.yiibai.com/docker/network_rm.html) | 删除一个或多个网络           |

- 