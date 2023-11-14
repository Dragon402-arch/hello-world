#### Milvus2.x版本操作

首先需要进入到 docker-compose 所在目录：`cd /data1/milvus2x`

目前版本：`Milvus v2.1.4 `

停止运行

```shell
 docker-compose down
```

后台启动

```shell
docker-compose up -d
```

查看状态

```shell
docker-compose ps
```





- 安装

  - 首先将两个配置文件存放到conf目录下

    - server_config.yaml
    - log_config.conf

  - 部署命令

    ```shell
    # 拉取镜像
    docker pull milvusdb/milvus:cpu-latest
    
    
    
    # 启动容器
    docker run -itd --name cpu-milvus -h milvus -p 19530:19530 -p 19121:19121 -p 9091:9091 -v /home/chinaoly/lis/milvus/db:/var/lib/milvus/db -v /home/chinaoly/lis/milvus/conf:/var/lib/milvus/conf -v /home/chinaoly/lis/milvus/logs:/var/lib/milvus/logs -v /home/chinaoly/lis/milvus/wal:/var/lib/milvus/wal milvusdb/milvus:cpu-latest
    
    
    
    docker run  -itd --name qdrant_db -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
    ```

- 使用

  - 连接测试

    ```python
    from pymilvus import connections
    connections.connect(
      alias="default", 
      host='192.168.51.59', 
      port='19530'
    )
    ```

    

  - 

