### 安装MySQL

- 获取安装文件

  - 官网下载：[下载链接](https://cdn.mysql.com/archives/mysql-8.0/mysql-8.0.28-1.el7.x86_64.rpm-bundle.tar)

  - 上传到服务器并解压：

    ```shell
    tar -xvf mysql-8.0.28-1.el7.x86_64.rpm-bundle.tar
    ```

  - 严格依次安装以下文件

    注意: 最好直接在 root 用户下操作，如果不是，则每个命令都要加上 sudo 

    ```shell
    # 查看CentOS系统有无安装 MariaDB数据库，如果有，则强制卸载，因为其和MySQL的安装会产生冲突。
    rpm -qa|grep mariadb
    
    # 强制卸载
    rpm -e --nodeps mariadb-libs
    
    sudo rpm -ivh mysql-community-common-8.0.28-1.el7.x86_64.rpm
    sudo rpm -ivh mysql-community-client-plugins-8.0.28-1.el7.x86_64.rpm
    sudo rpm -ivh mysql-community-libs-8.0.28-1.el7.x86_64.rpm
    sudo rpm -ivh mysql-community-client-8.0.28-1.el7.x86_64.rpm
    sudo rpm -ivh mysql-community-icu-data-files-8.0.28-1.el7.x86_64.rpm
    sudo rpm -ivh mysql-community-server-8.0.28-1.el7.x86_64.rpm
    
    # root用户下进行初始化
    mysqld --initialize --console
    
    # 此处的mysql不要修改
    chown -R mysql:mysql /var/lib/mysql/   
    ```

- 启动服务

  ```shell
  # 启动服务
  systemctl start mysqld
  
  # 查看root用户的临时密码以便进行初次登录
  cat /var/log/mysqld.log|grep localhost
  # 输出结果
  2022-09-18T08:18:58.801084Z 6 [Note] [MY-010454] [Server] A temporary password is generated for root@localhost: id+tk>Yo&4Pf
  
  # 登录,回车后输入临时密码。
  mysql -uroot -p
  
  # 修改临时密码之后才能进行查询操作。
  alter user 'root'@'localhost' identified by '123456';
  
  
  grant all on *.* to 'hive'@'localhost' identified by 'hive';
  
  # 验证
  show database;
  
  # 退出
  quit
  
  # 再次登录测试
  mysql -h 127.0.0.1 -uroot -p123456 
  
  # 查看连接端口
  show global variables like 'port';
  
  ```

- 操作服务（root用户下操作，或者命令前添加 sudo）

  ```shell
  # 启动服务
  systemctl  start mysqld
  
  # 终止服务
  systemctl  stop mysqld
  
  # 重启mysql服务
  systemctl restart mysqld
  ```

### 远程连接

- 远程连接

  ```sql
  # 切换到指定数据库
  use mysql
  
  # 新增用户
  create user 'lee'@'%' identified by 'ls145314';
  
  # 修改用户远程连接权限
  update user set host='%' where user='lee';
  
  # 修改用户认证方式
  alter user 'lee'@'%' identified with mysql_native_password by 'ls145314';
  
  
  alter user 'hive'@'%' identified with mysql_native_password by 'hive';
  
  # 刷新修改使其生效
  flush privileges;
  
  # 退出
  quit
  
  # 重启mysql服务使其生效
  systemctl restart mysqld
  ```

- 
