### 1. Docker 安装

### 2. Docker 配置

#### 2.1  修改镜像与容器存放目录

停止 docker 服务

```shell
service docker stop

systemctl stop docker.service
```

 修改配置文件

```shell
sudo vim /etc/docker/daemon.json

{
   "data-root": "/date/docker"
}

# 保存退出
```

配置信息更新（重新载入系统服务，必须执行！）

```shell
systemctl daemon-reload
```

原有文件移动到目标目录下

```shell
mv /var/lib/docker /date/
```

启动服务

```shell
service docker start

systemctl start docker.service
```

验证修改结果

```shell
docker info | grep "Docker Root Dir"

docker info

```