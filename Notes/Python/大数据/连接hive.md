集群控制中心登录：

```
URL:https://172.16.149.16:9022/mrscas/login?service=https%3A%2F%2F172.16.149.16%3A9022%2Fgateway%2Fmrsweb%2Fmrsmanager%2Fcas_security_check.htm#!/app/homepage/detail/summary

user_name/password: wx1140478-001/CQMYG@#14dss
```

使用 Python 连接 hive 共有三种方式，下面分别介绍三种连接方式：

### 1、使用 Pyhive 连接

- **依赖包：`pip install PyHive`**

- 连接注意事项：

  - 关于配置文件的问题

    - user.keytab ：存放位置随意
    - krb5.conf ：存放在 /etc/ 目录下

  - 关于 principal 的问题：

    ```shell
    # 报错信息：
    java.sql.SQLException: java.sql.SQLException: Could not open client transport with JDBC Uri: jdbc:hive2://192.168.100.56:10000/default;principal=wx1140478-002@67FA2087_E174_4577_9DED_1C4D0B0092EE.COM;auth=KERBEROS;sasl.qop=auth-conf: Kerberos principal should have 3 parts: wx1140478-002@67FA2087_E174_4577_9DED_1C4D0B0092EE.COM
    ```

    可知连接 hive 的 principal 需要包含三个部分： Kerberos principal should have 3 parts。

    ```shell
    # 新版华为云：
    kinit -kt /etc/qzyth/user.keytab wx1140478-002@67FA2087_E174_4577_9DED_1C4D0B0092EE.COM
    
    # 连接hive 需要使用的 principal 为：
    hive/hadoop.67fa2087_e174_4577_9ded_1c4d0b0092ee.com@67FA2087_E174_4577_9DED_1C4D0B0092EE.COM
    
    
    # 老版华为云：
    kinit -kt /opt/user.keytab chinaoly@HADOOP.COM
    
    # 连接hive 需要使用的 principal 为：
    hive/hadoop.hadoop.com@HADOOP.COM
    
    ```

    **连接时需要需要将 `hive/(……)@……` 括号中省略的部分追加到 /etc/hosts 文件内：**

    ```shell
    192.168.100.41 ClickHouseExts0001
    192.168.100.253 ClickHouseExts0002
    192.168.100.181 node-group-1xNmh0001
    192.168.100.114 node-group-1xNmh0002   
    192.168.100.38 node-group-1xNmh0003
    192.168.100.139 node-master3uhkv
    192.168.100.242 hadoop.67fa2087_e174_4577_9ded_1c4d0b0092ee.com node-master2oypn  # 关键所在
    192.168.100.56 67FA2087_E174_4577_9DED_1C4D0B0092EE.COM
    192.168.100.200 casserver
    ```

  - 关于 port 值：（在 hive-site.xml 文件中查找）

    ```xml
    -
    <property>
        <name>hive.server2.webui.port</name>
    
        <value>10002</value> # hive前端UI 的端口
    </property>
    
    -
    <property>
        <name>hive.server2.thrift.port</name> # hive 连接的端口
    
        <value>10000</value>
    </property>
    
    ```

- 连接代码示例：

  ```Python
  from pyhive import hive
  connection = hive.connect(
              host="192.168.100.242",  
              port=10000,
              database="default",
              auth="KERBEROS",
              kerberos_service_name="hive", # 作为 principal 的一部分拼接进去
          )
  cursor = connection.cursor()
  cursor.execute("show tables")
  for result in cursor.fetchall():
      print(result)
  ```

### 2、使用 impyla 连接

- **依赖包：`pip install impyla`**

- 关于配置文件的问题：同上

- 关于 principal 的问题：同上

- 代码连接示例：（该方式无须修改 hosts 文件，只需要填入 krb_host 参数即可）

  ```python
  from impala.dbapi import connect
  conn = connect(
      host="192.168.100.242",
      port=10000,
      database="default",
      auth_mechanism="GSSAPI",
      krb_host="hadoop.67fa2087_e174_4577_9ded_1c4d0b0092ee.com",
      kerberos_service_name="hive",
  )
  cursor = conn.cursor()
  cursor.execute("show tables")
  for result in cursor.fetchall():
      print(result)
  ```

### 3、使用 jaydebeapi 连接

- **依赖包：`pip install Jaydebeapi`**

- 关于配置文件的问题：同上

- 关于 principal 的问题：同上

- 配置依赖项：

  - jdbc 文件夹内的诸多 jar 包，存放在目标服务器上，如存放路径为：`/root/pythonProject/jdbc/`

  - 添加环境变量：

    ```shell
    vi ~/.bashrc
    
    # 追加一句
    export CLASSPATH=$CLASSPATH:/root/pythonProject/jdbc/*
    
    source ~/.bashrc
    ```

  若未配置依赖项则会出现该异常： `TypeError: Class org.apache.hive.jdbc.HiveDriver is not found`

- 代码连接示例：（该方式无须修改 hosts 文件）

  ```python
  import jaydebeapi 
  
  database = 'default'
  driver = 'org.apache.hive.jdbc.HiveDriver'
  host = '192.168.100.56'
  port = 10000
  principal = 'hive/hadoop.67fa2087_e174_4577_9ded_1c4d0b0092ee.com@67FA2087_E174_4577_9DED_1C4D0B0092EE.COM'
  
  url = f"jdbc:hive2://{host}:{str(port)}/{database};principal={principal};"
  
  
  # Connect to HiveServer2
  conn = jaydebeapi.connect(driver, url)
  cursor = conn.cursor()
  
  # Execute SQL query
  sql = "show tables"
  cursor.execute(sql)
  results = cursor.fetchall()
  print(results)
  conn.close()
  ```

- [代码连接示例来源](https://dwgeek.com/steps-to-connect-hiveserver2-from-python-using-hive-jdbc-drivers.html/)



