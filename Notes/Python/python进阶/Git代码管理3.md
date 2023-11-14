If you create a Git repository based on local sources, you need to add a remote repository to be able to collaborate on your Git project, as well as to eliminate the risks of storing all of your codebase locally. You **`push changes to a remote repository`** when you need to share your work and **`pull data from it`** to integrate changes made by other contributors into your local repository version.

如果你基于本地资源创建了一个Git 资源库，你需要为该Git 资源库（本地）添加一个远程资源库以便与人进行协作和消除将代码存放在本地资源库的风险。当你需要与他人共享的你的代码成果时，你会推送所有变化到一个远程资源库，当你需要将其他贡献者的所有变化融合到你的本地资源库时，你需要从远程资源库拖拉数据。

**If you have clone a remote Git repository,** for example from GitHub, **the remote is configured automatically and you do not have to specify it when you want to sync with it.** **The default name Git gives to the remote you've cloned from is *origin*.**

如果你克隆了一个远程 Git资源库，这个远程资源库会自动配置且当你想要本地和远程同步时，你不必指定它，也就是说进行push操作时，可以自动确定需要进行同步的远程资源库对象。Git将所克隆的远程资源库默认命名为 origin。

- 创建一个本地资源库的两种方式：

  - 克隆

    ```shell
    git clone url
    ```

    可以自动确定需要进行同步的远程资源库。

  - 初始化

    ```shell
    # 创建项目文件夹
    mdkir projectFloder
    
    # 切换目录，进入项目文件夹
    cd projectFloder
    
    # 初始化，开始版本控制
    git init
    ```

  - fetch 只是获取远程上更新的数据，但是并没有集成到本地资源库中，需要进行合并之后才是真正的融合。

  - update

    - update branch：如果需要将特定分支与其远程跟踪分支同步，请使用 update。这是获取并随后将更改应用于所选分支的快捷方式。PyCharm will pull changes from the remote branch and will rebase or merge them into the local branch.

      PyCharm中的位置：从主菜单中选择 **Git |Branch…**，点击后跳出一个菜单，可以针对某个分支进行update。

      ![image-20211228200836588](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20211228200836588.png)

    - update project ：If you have several project roots, or want to **fetch changes from all branches each time** you sync with the remote repository, you may find ***updating* your project** a more convenient option.When you perform the *update* operation, PyCharm fetches changes from all project roots and branches, and merge the tracked remote branches into your local working copy (equivalent to *pull*).

      

  - pull 

    - 如果需要从另一个分支而不是远程跟踪分支获取对当前分支的更改，请使用 pull。当您拉取时，您不仅可以下载新数据，还可以将其集成到项目的本地工作副本中。

    

  If you are more used to the concept of **staging changes for commit** instead of **using changelists where modified files are staged automatically**, select the **Enable staging area** option on the **Version Control | Git** page of the IDE settings.

  ![image-20211228203754666](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20211228203754666.png)

​		

Before pushing your changes, [sync with the remote](https://www.jetbrains.com/help/pycharm/2021.2/sync-with-a-remote-repository.html) and make sure your local copy of the repository is up-to-date to avoid conflicts.

红色， 表示在工作区(新建一个文件默认是红色的，添加到git之后会变成绿色的)

绿色， 表示在暂存区（PyCharm中默认是自动将文件添加到缓存区的，）

蓝色， 表示文件有修改，位于暂存区

文件名无颜色，表示位于本地仓库区或已经提交到远程仓库区

![image-20211228205116887](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20211228205116887.png)

