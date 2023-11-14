## vscode远程开发

##### 添加新服务器

![vscode截图](D:\Typora\Python\vscode截图.png)

- 先点击1中的齿轮，然后点击2中的配置文件，在配置文件中进行配置。

- 在配置文件中输入如下内容

  ```tex
  Host 59
      HostName 192.168.51.59
      User chinaoly
  ```

- 至此即可将连接该远程服务器。

##### 免密登录操作

- 首先在服务器用户的home目录下，查看是否存在 `.ssh` 的文件夹，若存在无须操作，若不存在则执行 **`ssh-keygen`** 该命令。

- 在本地Windows的命令行执行如下命令：

  ```shell
  ssh-keygen -t rsa -b 4096 -C "lis@chinaoly.com"
  
  ssh-keygen -t rsa -C "lis@chinaoly.com"
  
  C:\Users\千江映月\.ssh
  ```
  
  - `-t` 即指定密钥的类型（type），密钥的类型有两种，一种是`RSA`，一种是`DSA`
  
  - `-b` 指定密钥长度（bit缩写）。
  
  - `-C `表示提供一个注释（comment），用于识别这个密钥，可以省略。
  
  执行完毕之后会生成一个**.ssh** 文件夹，然后在外部新建一个名为`authorized_keys`的文件，然后将`.ssh`文件夹下**`id_rsa.pub`** 文件的内容复制到该文件中，然后使用如下命令上传至 服务器的 `.ssh` 文件下
  
  ```shell
  # 复制公钥，可以直接在外部进行粘贴
  clip < \Users\千江映月\.ssh\id_rsa.pub  
  
  # 上传到远程
  scp D:\authorized_keys  chinaoly@192.168.51.58:/home/chinaoly/.ssh/
  ```

​	执行完毕之后，再次连接服务器时就不必输入密码了。

- 若要将多台电脑的vscode都免密连接到服务器的一个用户下，则可以在将多个电脑的公钥都写入`authorized_keys` 文件中，一个公钥占用一行，换行书写另一个公钥。



