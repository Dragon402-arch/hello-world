- Docker-Compose将所管理的容器分为三层，

  - 项目（project）
  - 服务（service）
  - 容器（container）

- 命令

  ```shell
  docker-compose -h
  
  Commands:
    build       Build or rebuild services
    convert     Converts the compose file to platform's canonical format
    cp          Copy files/folders between a service container and the local filesystem
    create      Creates containers for a service.
    down        Stop and remove containers, networks
    events      Receive real time events from containers.
    exec        Execute a command in a running container.
    images      List images used by the created containers
    kill        Force stop service containers.
    logs        View output from containers
    ls          List running compose projects
    pause       Pause services
    port        Print the public port for a port binding.
    ps          List containers
    pull        Pull service images
    push        Push service images
    restart     Restart service containers
    rm          Removes stopped service containers
    run         Run a one-off command on a service.
    start       Start services
    stop        Stop services
    top         Display the running processes
    unpause     Unpause services
    up          Create and start containers
    version     Show the Docker Compose version information
  
  ```

- [参考](https://yeasy.gitbook.io/docker_practice/compose/commands)

- 启动项目

  将尝试自动完成包括构建镜像，（重新）创建服务，启动服务，并关联服务相关容器的一系列操作。

  ```shell
  docker-compose up
  
  # 后台启动项目
  docker-compose up -d
  
  # 在后台所有启动服务,指定编排文件docker-compose.yml
  
  docker-compose -f docker-compose.yml up -d
  ```

  `--force-recreate` 强制重新创建容器，不能与 `--no-recreate` 同时使用。

  `--no-recreate` 如果容器已经存在了，则不重新创建，不能与 `--force-recreate` 同时使用。

- 停止正在运行的项目服务

  ```shell
  docker-compose stop
  ```

- 启动停止运行的项目服务

  ```shell
  docker-compose start
  
  # 重启项目中的服务
  docker-compose restart
  ```

- 停止 `up` 命令所启动的容器，并移除网络

  ```shell
  #  Stop and remove containers, networks 默认删除容器和网络
  
  docker-compose down 
  
  ```

- 查看日志

  ```shell
  docker-compose logs 
  
  # 查看整个项目的日志
  docker-compose logs -t --tail="100"
  
  # 查看指定服务的日志
  docker-compose logs -t --tail="100" redis
  ```

- 列出项目中目前的所有容器

  ```shell
  docker-compose ps
  ```

- 列出 docker-compose 文件中包含的镜像

  ```shell
  docker-compose images
  ```

- 列出正在运行的所有项目

  ```shell
  docker-compose ls
  ```

- 删除所有（停止状态的）服务容器

  ```shell
  docker-compose rm
  ```

  