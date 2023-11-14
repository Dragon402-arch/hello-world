#### 导出和安装虚拟环境

- 导出虚拟环境

  ```shell
  # 将环境中安装的包写入requirements文件中
  pip freeze > requirements.txt 
  
  
  # 查看requirements文件中包含的package
  type requirements.txt
  ```

- 安装虚拟环境依赖包

  ```shell
  # 安装requirements文件中的包
  pip install -r requirements.txt
  ```

#### 安装Python

- 登录云服务器，创建新用户

  ```shell
  # 添加用户名
  adduser lee
  # 设置密码
  passwd lee
  
  # 提示两次输入新密码
  
  # 添加可写权限
  chmod 640 /etc/sudoers
  
  # 编辑文件
  vi /etc/sudoers
  
  # 添加一行
  lee     ALL=(ALL)       ALL
  
  # 改为只读权限
  chmod 440 /etc/sudoers
  ```

- 传输Python 安装包和配置环境

  ```shell
  /home/lee/
  
  scp -r E:/Python-3.7.10.tgz lee@124.70.134.105:/home/lee/
  
  
  # 进入到解压后的Python-3.7.10目录下，执行下面三个命令
  
  ./configure --prefix=/root/training/Python-3.7.10
  
  ./configure --prefix=/home/lee/Python3
  
  
  ./configure --prefix=/root/Python-3.7.10 --enable-optimizations --with-openssl=/usr/lib/ssl
  
  # --prefix是Python的安装目录。
  make
  
  make install
  
  # 至此安装完毕
  
  ```

  

- 替换依赖包

  ```shell
  mv /home/lee/myProject/torch_env/lib/ ./
  ```

  

  

  