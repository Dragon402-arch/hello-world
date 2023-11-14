- [参考](https://segmentfault.com/a/1190000040062218)

- 下载安装包：[地址](https://mirrors.tuna.tsinghua.edu.cn/apache/hive/hive-2.3.9/)

- 上传并解压

  ```shell
  sudo tar -zxvf apache-hive-2.3.9-bin.tar.gz -C /opt/
  ```

- 修改环境变量

  ```shell
  sudo vim ~/.bashrc
  
  export HADOOP_HOME=/opt/hadoop-2.7.1/
  export HIVE_HOME=/opt/apache-hive-2.3.9-bin
  export PATH=$HIVE_HOME/bin:$PATH
  
  # 使环境变量生效
  source ~/.bashrc
  ```

- 由于Hive需要将元数据，存储到Mysql中，因此需要添加 mysql-connector-java-8.0.28.jar 依赖包。

  ```shell
  sudo cp /home/hadoop/BigData/mysql-connector-java-8.0.28.jar /opt/apache-hive-2.3.9-bin/lib/
  
  sudo chown -R hadoop:hadoop  /opt/apache-hive-2.3.9-bin/
  
  ```
  
- 创建和修改hive的配置文件

  ```shell
  # 进入 hive 配置文件所在目录
  cd /opt/apache-hive-2.3.9-bin/conf/
  
  # 新建 hive-default.xml
  cp hive-default.xml.template hive-default.xml
  
  # 新建 hive-site.xml 文件
  vim hive-site.xml
  ```

  添加如下配置项：

  ```xml
  
  <configuration>
       <property>
           <name>javax.jdo.option.ConnectionURL</name>
          <value>jdbc:mysql://master:3306/hive?createDatabaseIfNotExist=true</value>
           <description>JDBC connect string for a JDBC metastore</description>
       </property>
       <property>
          #javax.jdo.option.ConnectionDriverName：连接数据库的驱动包。
          <name>javax.jdo.option.ConnectionDriverName</name>
          <value>com.mysql.jdbc.Driver</value>
           <description>Driver class name for a JDBC metastore</description>
       </property>
       <property>
          #javax.jdo.option.ConnectionUserName：数据库用户名。
          <name>javax.jdo.option.ConnectionUserName</name>
          <value>hive</value>
           <description>username to use against metastore database</description>
       </property>
       <property>
          #javax.jdo.option.ConnectionPassword：连接数据库的密码。
           <name>javax.jdo.option.ConnectionPassword</name>
           <value>hive</value>
           <description>password to use against metastore database</description>
       </property>
      <property>
          <name>hive.server2.thrift.port</name>
           <value>10000</value>
      </property>
      <property>
           <name>hive.server2.thrift.bind.host</name>
           <value>127.0.0.1</value>
      </property>
  </configuration>
  
  ```

  创建 hive-env.sh 文件

  ```shell
  cp hive-env.sh.template hive-env.sh
  
  vim hive-env.sh
  
  # 添加
  HADOOP_HOME=/opt/hadoop-2.7.1/
  export HIVE_CONF_DIR=/opt/apache-hive-2.3.9-bin/conf
  
  ```

- 启动 mysql 服务，创建用户以及数据库

  ```shell
  # 启动服务
  sudo systemctl  start mysqld
  
  # 登录
  mysql -u root -p 123456
  
  # 这个hive数据库与hive-site.xml中master:3306/hive的hive对应，用来保存hive元数据
  mysql> create database hive;
  
  ```

  配置 mysql 允许hive 接入

  ```shell
  
  use mysql;
  
  # 新增用户
  create user 'hive'@'%' identified by 'hive';
  
  # 修改用户远程连接权限
  update user set host='%' where user='hive';
  
  # 修改用户认证方式,末尾的hive是配置hive-site.xml中配置的连接密码
  alter user 'hive'@'%' identified with mysql_native_password by 'hive';
  
  # 将所有数据库的所有表的所有权限赋给hive用户，
  GRANT ALL PRIVILEGES ON *.* TO 'hive'@'%';
  
  # 刷新修改使其生效
  flush privileges;
  ```

  

- 启动 hive

  启动hive之前，请先启动hadoop集群。

  ```shell
  # 启动Hadoop的HDFS
  start-dfs.sh
  
  # 启动 yarn 
  start-yarn.sh
  
  # 启动 hive
  hive
  
  # 测试能否正常使用
  hive>show databases;  
  
  # 输出
  OK
  default
  Time taken: 10.599 seconds, Fetched: 1 row(s)
  
  # 退出
  hive>exit;
  
  ```

  启动报错：

  ```shell
  FAILED: SemanticException org.apache.hadoop.hive.ql.metadata.HiveException:java.lang.RuntimeException: Unable to instantiate org.apache.hadoop.hive.ql.metadata.SessionHiveMetaStoreClient
  ```

  解决办法：

  ```shell
  cd  /opt/apache-hive-2.3.9-bin/bin/
  
  # 执行如下命令后再启动hive，即可正常启动。仅在初次启动时执行，之后的启动不需要再次执行。
  schematool -dbType mysql -initSchema
  ```

  

