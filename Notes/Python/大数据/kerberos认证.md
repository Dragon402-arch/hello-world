- 两种认证情形：

  - Authenticate both the server and the client using Kerberos

  - Authenticate only the client using Kerberos

    `cat /etc/ssh2/sshd2_config` 若 `AllowedAuthentications=gssapi-keyex`则进行双向认证;若 `AllowedAuthentications=gssapi-with-mic` 则仅进行客户端认证。

- 安装依赖包

  - 在线安装

    ```shell
    sudo yum install krb5-server krb5-libs krb5-workstation
    
    sudo yum -y install gcc libkrb5-dev
    
    sudo yum install -y krb5-devel
    
    sudo yum install gcc-c++
    sudo yum install cyrus-sasl-devel
    sudo yum install cyrus-sasl-gssapi
    
    
    # 安装完成上面的依赖以后才能安装成功 gssapi 包
    pip install gssapi
    
    # 安装成功这个依赖以后才能成功安装 sasl 包
    yum install gcc-c++
    yum install cyrus-sasl-devel
    yum install cyrus-sasl-gssapi
    yum install unixODBC-devel
    
    
    pip install sasl
    ```

    

  - 离线安装

    ```shell
    yum install -y krb5-devel --skip-broken
    
    yum install libkadmin
    
    yum install krb5-server
    
    yum install krb5-libs 
    
    yum install krb5-workstation
    
    yum install cyrus-sasl-gssapi
    
    yum install cyrus-sasl
    
    yum install cyrus-sasl-devel
    
    
     /dev/sda
    ```

    

  

  

- 编辑和修改配置文件

  ```shell
  vi /etc/krb5.conf
  
  ```

  配置文件初始内容

  ```
  [logging]
   default = FILE:/var/log/krb5libs.log
   kdc = FILE:/var/log/krb5kdc.log
   admin_server = FILE:/var/log/kadmind.log
  [libdefaults]
   dns_lookup_realm = false
   ticket_lifetime = 24h
   renew_lifetime = 7d
   forwardable = true
   rdns = false
   pkinit_anchors = /etc/pki/tls/certs/ca-budle.crt
   # default_realm = EXAMPLE.COM
   default_ccache_name = KEYRING:persistent:%{uid}
  [realms]
  # EXAMPLE.COM = {
  # kdc = kerberos.example.com
  # admin_server = kerberos.example.com
  # }
  [domin_realm]
   #.example.com = EXAMPLE.COM
   # example.com = EXAMPLE.COM
  
  ```

  修改后的配置文件内容

  ```
  [logging]
   default = FILE:/var/log/krb5libs.log
   kdc = FILE:/var/log/krb5kdc.log
   admin_server = FILE:/var/log/kadmind.log
  [libdefaults]
   dns_lookup_realm = false
   ticket_lifetime = 24h
   renew_lifetime = 7d
   forwardable = true
   rdns = false
   pkinit_anchors = /etc/pki/tls/certs/ca-budle.crt
   default_realm = HADOOP.COM
   default_ccache_name = KEYRING:persistent:%{uid}
  [realms]
   HADOOP.COM = {  # 两个均修改为 hostname -f 的值
   kdc = localhost
   admin_server = localhost
  }
  [domin_realm]
   .hadoop.com = HADOOP.COM
   hadoop.com = HADOOP.COM
  
  ```

- 初始化Kerberos数据库

  ```shell
  krb5_util create -s
  ```

  输出：

  ![image-20220625100835715](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20220625100835715.png)

  提示：需要将圈着的部分使用键盘输入两次。

- 启动服务

  ```shell
  systemctl start krb5kdc
  systemctl start kadmin
  
  # 或
  service krb5kdc start
  service kadmin start
  
  # 创建symlink
  systemctl enable krb5kdc
  systemctl enable kadmin
  
  
  ```

- 开始认证（authenticate）

  ```shell
  kadmin.local
  
  # 该命令与下面的分布操作命令作用相同
  kadmin.local -q "addprinc admin/admin"
  ```

  创建管理员：

  ![image-20220625102402541](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20220625102402541.png)

  完成上述操作之后，修改一个文件的配置

  ```shell
  vi /var/kerberos/krb5kdc/kadm5.acl
  
  # 原文件内容
  */admin@EXAMPLE.COM     *
  # 修改为：
  */admin@HADOOP.COM     *
  
  # systemctl restart kadmin
  
  service kadmin restart
  service krb5kdc restart
  ```

  说明：`kadmin`和`kadmin.local`都是 KDB 的管理接口，区别在于`kadmin.local`只能在 Server 上使用，无需密码；`kadmin`在 Server 和 Client 上都能使用，需要密码。（首先需要在 Server 上启动 Kadmin 服务，才能在 Client 上进行使用）

  此时在以管理员身份再添加新的凭证（credential）

  ![image-20220625105614696](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20220625105614696.png)

  ```shell
  # 初次进入 kadmin 时使用 kadmin.local 添加管理员admin/admin 用户之后，使用如下命令进入。
  kadmin -p admin/admin@HADOOP.COM -w password
  
  # 进入之后不再是显示 kadmin.local: 而是 kadmin: 
  
  # 添加新用户
  addprinc abcd
  
  # Principal "abcd@HADOOP.COM" created
  
  # 在输入两次密码之后，可以查看用户列表，发现有了新增的用户，使用下面命令可以查看
  
  listprincs
  
  
  ```

  [讲解](https://www.youtube.com/watch?v=YRLEapMDZmU)

- 为集群（cluster）中的每个服务（service）指定对应的 kerberos Principal ,

  ![image-20220625110146761](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20220625110146761.png)

  添加完成之后，进入kadmin，输入listprincs，可以发现每个服务都生成了对应的kerberos principal

  ![image-20220625110449513](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20220625110449513.png)

- 客户端连接

  - 第一步 ：将集群主机的krb5.conf文件复制过来替换掉客户端服务器上的 /etc/krb5.conf

  - 第二步：

    ```shell
    vi /etc/hosts
    
    # 将集群中节点的ip和域名（如下图所示） 添加到 /etc/hosts 文件中
    192.168.52.38 cbp1.chinaoly.com
    192.168.52.37 cbp2.chinaoly.com
    192.168.52.36 cbp3.chinaoly.com
    192.168.52.35 cbp4.chinaoly.com
    192.168.52.34 cbp5.chinaoly.com
    192.168.52.33 cbp6.chinaoly.com
    192.168.52.32 cbp7.chinaoly.com
    192.168.52.31 cbp8.chinaoly.com
    192.168.52.39 cbp9.chinaoly.com
    192.168.52.42 cbp10.chinaoly.com
    192.168.52.43 cbp11.chinaoly.com
    192.168.52.44 cbp12.chinaoly.com
    192.168.52.45 cbp13.chinaoly.com
    
    
    
    192.168.52.175 192-168-52-175 192-168-52-175.
    192.168.52.176 192-168-52-176 192-168-52-176.
    192.168.52.174 192-168-52-174 192-168-52-174.
    192.168.52.173 192-168-52-173 192-168-52-173.
    192.168.52.171 192-168-52-171 192-168-52-171.
    192.168.52.172 192-168-52-172 192-168-52-172.
    
    
    192.168.52.175 192-168-52-175 192-168-52-175.
    192.168.52.176 192-168-52-176 192-168-52-176.
    192.168.52.174 192-168-52-174 192-168-52-174.
    192.168.52.173 192-168-52-173 192-168-52-173.
    192.168.52.171 192-168-52-171 192-168-52-171.
    192.168.52.172 192-168-52-172 192-168-52-172.
    
    
    # 连接hive的语句
    jdbc:hive2://cbp5.chinaoly.com:10000/;principal=hive/cbp5.chinaoly.com@CHINAOLY.COM
    
    # 当访问集群中某个节点的ip时，会映射到指定的域名，并将域名与其他参数进行组合构成一个principal，如果该principal存在于listprincs中，则可以进行正常连接，否则就会认证失败。
    
    ```

  - 第三步：请求票据，进行认证

    ```shell
    # 请求票据
    kinit -kt /etc/user.keytab chinaoly@HADOOP.COM
    # 其中 keytab 文件记录了一个或者多个Kerberos用户与密码信息。
    
    # 查看本地缓存的票据
    klist
    
    
    Major (851968): Unspecified GSS failure.Minor code may provide more information,Minor(2529639053)No Kerberos credentials available (default cache: KEYRING:persistent:0
    
    klist credentials cache: KEYRING:persistent:0 not found
    
    
    kinit -kt  /opt/conf/user.keytab zhongao@HADHOOP.COM
    ```

    

  - 认证测试

    ```shell
    # 连接测试
    curl -XGET --negotiate --tlsv1.2 -k -u : "https://192.168.52.174:24100/"
    
    
    beeline -u "jdbc:hive2://192.168.52.173:21066/;principal=hive/hadoop.hadoop.com@HADOOP.COM"
    
    curl -XGET : "https://192.168.51.43:9286/"
    
    
    {
      "name" : "EsNode1@192.168.52.174",
      "cluster_name" : "elasticsearch_cluster",
      "cluster_uuid" : "CLymmZvoQq-9tS-voQml0g",
      "version" : {
        "number" : "6.7.1",
        "build_flavor" : "default",
        "build_type" : "tar",
        "build_hash" : "Unknown",
        "build_date" : "2019-07-09T19:36:19.015821Z",
        "build_snapshot" : false,
        "lucene_version" : "7.7.0",
        "minimum_wire_compatibility_version" : "5.6.0",
        "minimum_index_compatibility_version" : "5.0.0"
      },
      "tagline" : "You Know, for Search"
    }
    
    # 若输出以上内容表示连接正常
    ```

    curl 版本的查询命令

    ```shell
    curl -X GET --negotiate --tlsv1.2 -k -u: "https://192.168.52.174:24100/sub_sj_es_lxdhztb/_search" -H 'Content-Type: application/json' -d'
    {
        "query": {
            "match_all": {}
        }
    }
    '
    ```

    

    curl版本的查看索引状态

    ```shell
     curl -XGET --negotiate --tlsv1.2 -k -u : "https://192.168.52.174:24100/_cat/indices?v"
    ```

    

  - 

  

- https://www.youtube.com/watch?v=7Q-Xx0I8PXc


