- [教程](https://realpython.com/python-git-github-intro/)
  
- 代码上传
  
  - 操作顺序
  
    1、先将服务器上的代码下载到本地Pycharm，更新本地代码，
  
    2、update 远程资源库上的代码到本地，更新本地代码（有时需要选择保留本地修改，还是远程修改）
  
    3、然后 commit and push
  
- 版本控制

  - **Git 分布式版本控制系统**

    - `Working Directory`
    - `Stage Environment`
      - milestone 里程碑
    - `History`
    - `Remote Directory`

  - SVN （SubVersion）集中式版本控制系统

    - 三个阶段：

      - from  `local` to  `stage` 
      - from `stage` to `commit` 
    
    - git 命令：
    
      **配置Git**

      ```shell
      git --version
      git config --global user.name "w3schools-test"
      git config --global user.email "test@w3schools.com"
      ```
      
      Use `global` to set the **username and e-mail** for **every repository** on your computer.
      
      If you want to set the username/e-mail for just **the current repo**, you can remove `global`
      
      **创建一个文件夹，并在文件夹内进行Git初始化。**
      
      ```shell
      git init
      # Initialized empty Git repository in D:/mygitproject/.git/
      
      
      git clone url
      ```
      
      **创建文件**
      
      ```shell
      touch index.html
      ```
      
      
      
      **Git 状态**
      
      ```shell
      # 查看Git状态
      git status
      
      # 输出结果：
      On branch master
      
      No commits yet
      
      Untracked files:
        (use "git add <file>..." to include in what will be committed)
              index.html.
      
      nothing added to commit but untracked files present (use "git add" to track)
      
      ```
      
      **将文件添加到缓存区并查看Git状态**
      
      ```shell
      # 上传至缓存区
      git add index.html
       
      # 查看状态
      git status
      
      # 输出结果：
      On branch master
      
      No commits yet
      
      Changes to be committed:
        (use "git rm --cached <file>..." to unstage)
              new file:   index.html
      ```
      
      **创建多个文件，一次添加多个文件到缓存区**
      
      ```shell
      git add --all # 或是 git add -A
      
      # Using --all instead of individual filenames will stage all changes (new, modified, and deleted) files.
      
      # 使用 --all 而不是单个的文件名称，将会使得所有的变化（新建，修改，删除）到添加到缓存区。
      ```
      
      **将缓存区的文件提交（commit）**
      
      When we `commit`, we should **always** include a **message**.
      
      The `commit` command performs a commit, and the `-m "message"` adds a message.By adding clear messages to each `commit`, it is easy for yourself (and others) to see what has changed and when.
      
      ```shell
      git commit -m "First release of Hello World!"
      ```
      
      将修改后的文件直接提交，而不是先添加到缓存区。比如修改了index.html文件
      
      ```shell
      git status --short
      
      # 输出：M index.html
      ```
      
      Short status flags are:
      
      - `??` - Untracked files
      - `A` - Files added to stage
      - `M` - Modified files
      - `D` - Deleted files
      
      Sometimes, when you make small changes, using the staging environment seems like a waste of time. It is possible to commit changes directly, skipping the staging environment. The `-a` option will automatically stage every changed, already tracked file.
      
      ```shell
      # 跳过缓存区，直接提交。
      git commit -a -m "Updated index.html with a new line"
      
      ```
      
       **Skipping the Staging Environment is not generally recommended**.Skipping the stage step can sometimes make you include unwanted changes.
      
      若要查看一个项目仓库的历史提交记录，可以使用如下命令：
      
      ```shell
      git log
      ```
      
      使用帮助
      
      ```shell
      # 查看给定命令
      git command -help  # See all the available options for the specific command
      git commit -help
      
      # 查看所有命令
      git help --all #  See all possible commands
      ```
      
      In Git, a `branch` is a new/separate version of the main repository.
      
      - 主分支 main branch
      - 新分支 new branch
      
      **创建新分支**
      
      ```shell
      git branch hello-world-images
      ```
      
      **查看项目分支**
      
      ```shell
      git branch
      
      # 输出：
        hello-world-images
      * master  # * 表示当前处于该分支(主分支)上
      
      # 查看所有远程分支
      git branch -r
      ```
      
      **移动到新分支**
      
      ```shell
      
      git checkout hello-world-images  # 移动到新分支的命令
      Switched to branch 'hello-world-images' #  输出结果
      
      # 可以看到在移动到新分支之后，项目文件后边的分支名称也发生了改变,由master变为hello-world-images。
      ```
      
      在新分支下，在项目文件中添加一个 `jery.jpeg`的图片，然后修改`index.html`文件，
      
      **查看新分支下Git状态：**
      
      ```shell
      git status
      
      # 输出：
      On branch hello-world-images
      Changes not staged for commit:
        (use "git add <file>..." to update what will be committed)
        (use "git restore <file>..." to discard changes in working directory)
              modified:   index.html
      
      Untracked files:
        (use "git add <file>..." to include in what will be committed)
              jery.jpeg
      
      
      ```
      
      可以看到`index.html`发生了修改，以及一个新增的文件 `jery.jpeg`。
      
      上传所有文件到缓存区
      
      ```shell
      git add --all
      
      # 查看状态
      git status
      
      On branch hello-world-images
      Changes to be committed:
        (use "git restore --staged ..." to unstage)
          new file: jery.jpeg
          modified: index.html
      
      # 提交
      git commit -m "Added image to Hello World"
      ```
      
      查看当前分支下的文件
      
      ```shell
      ls
      
      # 输出：README.md  bluestyle.css  index.html  jery.jpeg
      
      # 切换分支
      git checkout master
      
      # 查看主分支下的文件
      ls
      
      # 输出：README.md  bluestyle.css  index.html
      # index.html文件中的内容仍然是修改前的内容，而新分支下的该文件则保存的是修改后的内容。
      ```
      
      **若分支不存在，则创建分支并移动到新创建的分支，若存在则直接移动到该分支。**
      
      Using the `-b` option on `checkout` will create a new branch, and move to it, if it does not exist
      
      ```shell
      git checkout -b emergency-fix
      ```
      
      新创建的分支会包含主分支下的所有文件，然后可以这些文件进行操作，也可以创建新的文件。
      
      **合并分支**
      
      切换到主分支下，然后合并分支：
      
      ```shell
      # 切换分支
      git checkout master
      
      # 合并分支
      git merge emergency-fix
      
      # 删除分支
      git branch -d emergency-fix
      ```
      
      下载
      
      - curl url 下载资源
      - git clone url  获取项目文件
      
      

​	
