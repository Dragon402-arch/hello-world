- 使用镜像创建MySQL数据库

  ```shell
  # mysql下载镜像
  docker pull mysql:8.0.28
  
  # 使用Docker run 创建容器，并且做好端口映射和目录挂载。
  docker run -d --name=m1 --net mynet  --ip 172.18.0.3 -p 4306:3306 -v /root/mysql:/var/lib/mysql -e MYSQL_ROOT_PASSWORD=abc123456 mysql:8.0.28
  ```

  

- Docker Compose 使用

