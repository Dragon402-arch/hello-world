### Docker Compose 

#### 1.1 引入与简介

- 引入
  - 为了完成一个完整的项目，需要使用多容器完成业务开发，容器之间会存在依赖关系，被依赖的容器要优先启动，容器启动的先后顺序至关重要。
  - 一组容器没有划归到一个项目下面，进行多服务器部署存在困难。
- 简介
  - 以项目为对象，将项目中一组相关联的容器整合在一起，对这组容器进行编排按照指定顺序启动。
  - 应用场景为多容器项目。

#### 1.2 安装与卸载

- 安装

  - 在线安装

    ```shell
    sudo curl -L https://github.com/docker/compose/releases/download/v2.11.2/docker-compose-linux-x86_64 > /usr/local/bin/docker-compose
    
    sudo chmod +x /usr/local/bin/docker-compose
    
    # 测试
    
    docker-compose -v
    ```

    

  - 离线安装

    ```shell
    # 下载并上传 docker-compose-linux-x86_64 文件
    mv docker-compose-linux-x86_64 docker-compose
    
    mv ./docker-compose /usr/local/bin
    
    sudo chmod +x /usr/local/bin/docker-compose
    
    
    https://docs.docker.com/compose/compose-file/compose-file-v3/
    ```

- 卸载

  删除 `/usr/local/bin` 下面的 `docker-compose`即可。

#### 1.3 使用

- docker-compose.yml 文件示例

  ```yaml
  
  version: "3.9"
  services:
  
    redis:
      image: redis:alpine
      ports:
        - "6379"
      networks:
        - frontend
      deploy:
        replicas: 2
        update_config:
          parallelism: 2
          delay: 10s
        restart_policy:
          condition: on-failure
  
    db:
      image: postgres:9.4
      volumes:
        - db-data:/var/lib/postgresql/data
      networks:
        - backend
      deploy:
        placement:
          max_replicas_per_node: 1
          constraints:
            - "node.role==manager"
  
    vote:
      image: dockersamples/examplevotingapp_vote:before
      ports:
        - "5000:80"
      networks:
        - frontend
      depends_on:
        - redis
      deploy:
        replicas: 2
        update_config:
          parallelism: 2
        restart_policy:
          condition: on-failure
  
    result:
      image: dockersamples/examplevotingapp_result:before
      ports:
        - "5001:80"
      networks:
        - backend
      depends_on:
        - db
      deploy:
        replicas: 1
        update_config:
          parallelism: 2
          delay: 10s
        restart_policy:
          condition: on-failure
  
    worker:
      image: dockersamples/examplevotingapp_worker
      networks:
        - frontend
        - backend
      deploy:
        mode: replicated
        replicas: 1
        labels: [APP=VOTING]
        restart_policy:
          condition: on-failure
          delay: 10s
          max_attempts: 3
          window: 120s
        placement:
          constraints:
            - "node.role==manager"
  
    visualizer:
      image: dockersamples/visualizer:stable
      ports:
        - "8080:8080"
      stop_grace_period: 1m30s
      volumes:
        - "/var/run/docker.sock:/var/run/docker.sock"
      deploy:
        placement:
          constraints:
            - "node.role==manager"
  
  networks:
    frontend:
    backend:
  
  volumes:
    db-data:
  ```

  ```yaml
  
  version: "3.9"
  services:
  
    mysqldb:
      image: mysql:5.7.19
      ports:
        - "3306:3306"
      container_name: test_mysql
      networks:
        - mynet
      volumes:
        - "/root/mysql/mysql_back_data:/var/lib/mysql"
      environment:
        - "MYSQL_ROOT_PASSWORD=root"
      depends_on:
        - redis
  
    redis:
      image: redis:4.0.14
      container_name: redis_test
      ports:
        - "6379:6379"
      networks:
        - mynet
      volumes:
        - "/root/redis/redis_backup_data:/data"
      command: "redis-sever"
  
  
  networks:
    mynet:
        
        
        
  ```

- 编写 docker-compose.yml 文件

  ```shell
  docker version
  
  # 或
  
  docker -v
  
  # 输出 Docker version 20.10.18, build b40c2f6
  # 根据该版本确定 yml 文件 中 version 的具体版本值。
  
  docker-compose up 
  
  
  # 数据卷存在的必要性是为了在容器删掉之后，可以使用宿主机上面的备份数据进行恢复
  
  bulid 是使用Dockerfile文件直接创建镜像，而image则是使用现有镜像，两者不能同时存在
  ```
  
  