### zookeeper 安装

- 下载安装包：[链接](https://archive.apache.org/dist/zookeeper/stable/apache-zookeeper-3.6.3-bin.tar.gz )

- 下载并解压

  ```shell
  wget https://archive.apache.org/dist/zookeeper/stable/apache-zookeeper-3.6.3-bin.tar.gz
  
  sudo tar -zxvf apache-zookeeper-3.6.3-bin.tar.gz -C /opt/
  ```

- 创建工作目录：

  ```shell
  cd /opt/apache-zookeeper-3.6.3-bin
  
  # 创建快照日志存放目录
  mkdir -p dataDir 
  
  # 创建事务日志存放目录
  mkdir -p dataLogDir
  
  ```

  `注意`:如果不配置dataLogDir，那么事务日志也会写在dataDir目录中。这样会严重影响zk的性能。因为在zk吞吐量很高的时候，产生的事务日志和快照日志太多。

- 修改zoo.cfg配置文件

  ```shell
  dataDir=/opt/apache-zookeeper-3.6.3-bin/dataDir
  dataLogDir=/opt/apache-zookeeper-3.6.3-bin/dataLogDir
  server.1=master:2888:3888
  server.2=node01:2888:3888
  server.3=node02:2888:3888
  ```

  dataDir指定的目录下面，创建一个myid文件,里面内容为一个数字，用来标识当前主机

  ```shell
  sudo chown -R hadoop /opt/apache-zookeeper-3.6.3-bin
  
  echo "1" > /opt/apache-zookeeper-3.6.3-bin/dataDir/myid
  echo "2" > /opt/apache-zookeeper-3.6.3-bin/dataDir/myid
  echo "3" > /opt/apache-zookeeper-3.6.3-bin/dataDir/myid
  
  # 将整个 apache-zookeeper-3.6.3-bin 传输到集群的每个节点，并修改myid文件的值为2和3
  sudo scp -rp apache-zookeeper-3.6.3-bin hadoop@192.168.56.106:/home/hadoop
  
  sudo scp -rp apache-zookeeper-3.6.3-bin hadoop@node02:/home/hadoop
  
  ```

  

- 添加环境变量

  ```shell
  vim ~/.bashrc
  
  export ZOOKEEPER_HOME=/opt/apache-zookeeper-3.6.3-bin
  export PATH=$ZOOKEEPER_HOME/bin:$PATH
  
  source ~/.bashrc 
  ```

  主节点上由于安装了其他插件，环境变量会在此基础上有所增加。

- 启动 zookeeper

  ```shell
  cd /opt/apache-zookeeper-3.6.3-bin/bin
  
  #启动ZK服务: 
  ./zkServer.sh start
  
  #停止ZK服务: 
  ./zkServer.sh stop
  
  #重启ZK服务: 
  ./zkServer.sh restart
  
  #查看ZK服务状态: 
  ./zkServer.sh status
  ```

  在Zookeeper集群的每个结点上，执行启动ZooKeeper服务的脚本:`./zkServer.sh start`,启动后执行jps若有线程QuorumPeerMain则执行成功。**Zookeeper集群需要每台挨个启动**。启动集群的时候，集群数量启动没有超过一半，状态会有错误提示，当集群启动数量超过一半就会自动转为正常状态，并且此台使集群进入正常工作状态的服务器会成为leader角色，集群中其他服务器的角色为follower。

  输出：

  ```shell
  [hadoop@master bin]$ jps
  9201 QuorumPeerMain
  9885 Jps
  
  [hadoop@master bin]$ ./zkServer.sh status
  ZooKeeper JMX enabled by default
  Using config: /opt/apache-zookeeper-3.6.3-bin/bin/../conf/zoo.cfg
  Client port found: 2181. Client address: localhost. Client SSL: false.
  Mode: follower
  
  [hadoop@node01 bin]$ ./zkServer.sh status
  ZooKeeper JMX enabled by default
  Using config: /opt/apache-zookeeper-3.6.3-bin/bin/../conf/zoo.cfg
  Client port found: 2181. Client address: localhost. Client SSL: false.
  Mode: follower
  
  [hadoop@node02 bin]$ ./zkServer.sh status
  /usr/bin/java
  ZooKeeper JMX enabled by default
  Using config: /opt/apache-zookeeper-3.6.3-bin/bin/../conf/zoo.cfg
  Client port found: 2181. Client address: localhost. Client SSL: false.
  Mode: leader
  ```

  

### hbase 安装

- 参考：
  - [参考1](https://blog.csdn.net/qq_39208832/article/details/118518522)
  - [参考2](https://blog.csdn.net/loveandstory/article/details/114077470)

- 下载安装包：[链接](http://archive.apache.org/dist/hbase/2.1.4/)

- 上传并解压

  ```shell
  sudo tar -zxvf hbase-2.1.4-bin.tar.gz -C /opt/
  ```

- 修改环境变量

  ```shell
  vim ~/.bashrc
  
  export JAVA_HOME=/usr/java/default
  export HBASE_HOME=/opt/hbase-2.1.4
  # 在原有环境变量的基础上追加上hbase的路径。
  export PATH=$PATH:$HIVE_HOME/bin:$HBASE_HOME/bin
  
  # 使环境变量生效
  source ~/.bashrc
  ```

  node01、node02节点添加

  ```shell
  vim ~/.bashrc
  
  export JAVA_HOME=/usr/java/default
  export HADOOP_HOME=/opt/hadoop-2.7.1/
  export ZOOKEEPER_HOME=/opt/apache-zookeeper-3.6.3-bin
  export HBASE_HOME=/opt/hbase-2.1.4
  export PATH=$ZOOKEEPER_HOME/bin:$HBASE_HOME/bin:$PATH
  
  source ~/.bashrc
  
  ```

  

- 修改配置文件

  - hbase-env.sh

    ```shell
    export JAVA_HOME=/usr/java/default
    #使用自己安装的zookeeper
    export HBASE_MANAGES_ZK=false
    
    ```

    

  - hbase-site.xml

    ```xml
    <configuration>
    <!--hbase.root.dir 将数据写入哪个目录 如果是单机版只要配置此属性就可以，
    value中file:/绝对路径，如果是分布式则配置与hadoop的core-site.sh服务器、端口以及zookeeper中事先创建的目录一致-->
    	<property>
         	<name>hbase.root.dir</name>
         	<value>hdfs://master:9000/hbase</value>
    	</property>
    
    <!--单机模式不需要配置，分布式配置此项,value值为true,多节点分布-->
    	<property>
         	<name>hbase.cluster.distributed</name>
         	<value>true</value>
    	</property>
    
    <!--单机模式不需要配置 分布式配置此项,value为zookeeper的conf下的zoo.cfg文件下指定的物理路径dataDir=/export/software/zookeeper3.6.2/dataDir-->
    	<property>
        	 <name>hbase.zookeeper.property.dataDir</name>
        	 <value>/opt/apache-zookeeper-3.6.3-bin/dataDir</value>
    	</property>
    <!--端口默认60000-->
    	<property>
         	<name>hbase.master.port</name>
         	<value>16000</value>
    	</property>
    
    <!--zookooper 服务启动的节点，只能为奇数个-->
    	<property>
         	<name>hbase.zookeeper.quorum</name>
         	<value>master,node01,node02</value>
    	</property>
    </configuration>
    
    
    ```

    

  - 配置regionservers

    ```shell
    vim regionservers
    
    master
    node01
    node02
    
    ```

    

  - 配置分发

    ```shell
    sudo scp -rp hbase-2.1.4 hadoop@node01:/home/hadoop
    sudo scp -rp hbase-2.1.4 hadoop@node02:/home/hadoop
    
    sudo mv hbase-2.1.4 /opt
    sudo chown -R hadoop /opt/hbase-2.1.4
    ```

    

- 配置NTP
   配置master为备选ntp时间同步服务器(当节点无法连接外网进行时间同步), slave1和slave2无法向外网同步时间时候, 向master看齐;

  当master也无法同步外网时间时,都使用master的本地时钟;( 如果只有master无法同步外网, 可能时间会不同步 =.=)

  在master节点配置：

  ```shell
  su root
  
  vim /etc/ntp.conf
  # 追加
  server 127.127.1.0 # local clock
  fudge 127.127.1.0 stratum 10
  
  systemctl start ntpd.service	//启动ntp服务
  
  ntpq -p      // 查看同步时间
  ```

   在node01和node02节点配置ntp

  ```shell
  su root
  
  vim /etc/ntp.conf
  # 追加
  server master
  
  systemctl restart ntpd
  
  ntpstat
  ```

- 先启动hadoop、zookeeper，

  ```shell
  # 启动Hadoop的HDFS
  start-dfs.sh
  
  # 启动 yarn 
  start-yarn.sh
  
  # 启动zookeeper
  cd /opt/apache-zookeeper-3.6.3-bin/bin
  
  #启动ZK服务: 
  ./zkServer.sh start
  
  #查看ZK服务状态: 
  ./zkServer.sh status
  
  #启动hbase，只在master上执行即可。
  cd /opt/hbase-2.1.4/bin/
  
  ./start-hbase.sh
  
  # 正常启动输出
  [hadoop@master bin]$ jps
  4432 HMaster
  4976 Jps
  3345 DataNode
  4562 HRegionServer
  3208 NameNode
  4105 QuorumPeerMain
  3610 ResourceManager
  3725 NodeManager
  
  [hadoop@node01 bin]$ jps
  3737 HRegionServer
  4025 Jps
  3179 DataNode
  3311 NodeManager
  3519 QuorumPeerMain
  
  [hadoop@node02 bin]$ jps
  3024 NodeManager
  2817 DataNode
  2930 SecondaryNameNode
  3764 Jps
  3212 QuorumPeerMain
  3453 HRegionServer
  
  
  ```

- 进入shell界面

  ```
  hbase shell
  
  hbase(main):001:0> list
  TABLE
  0 row(s)
  Took 0.7360 seconds
  => []
  
  
  hbase(main):001:0> create 'school:student',{NAME => 'essential'},{NAME => 'additional'}
  hbase(main):002:0> create 'school:teacher',{NAME => 'essential'},{NAME => 'additional'}
  hbase(main):003:0> list
  TABLE
  school:student
  school:teacher
  2 row(s)
  Took 0.0135 seconds
  => ["school:student", "school:teacher", "teacher"]
  
  ```

- 进入HBase Web 页面

  ```shell
  192.168.56.104:16010
  ```

  ![image-20220926093547430](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20220926093547430.png)

-  先关闭hbase集群再关闭hadoop！
