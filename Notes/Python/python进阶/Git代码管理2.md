- Push Local Repository to GitHub

  ```shell
  # 为本地资源库添加远程资源库
  git remote add origin https://github.com/w3schools-test/hello-world.git
  
  # origin 是默认的远程版本库名称，master是本地分支的名称，将当前分支与远程分支关联，初次推送时需要，再次推送直接git push即可。
  git push --set-upstream origin master
  ```

  - branch 分支 （pull的对象，merge的对象）

    - 分支操作

      ```shell
      # 创建分支
      git branch branch_name
      
      # 若分支存在则切换到该分支，否则创建之后再切换到该分支。
      git checkout -b branch_name 
      
      # 删除分支
      git branch -d branch_name
      ```

    - merge ：将当前分支与指定分支合并。

      - 将本地的两个分支合并

      - 将本地分支与远程分支合并

        ```shell
        # 将origin/master分支（远程分支）与当前分支合并
        git merge origin/master 
        ```

    - pull：`pull` is a combination of `fetch` and `merge`. It is used to pull all changes **from a remote repository into the branch** you are working on.拉取远程资源库的所有变化并融合到当前本地分支。

      ```shell
      git pull origin
      
      # 更新本地资源库
      git pull 
      
      # 将新的远程分支添加到本地
      git checkout html-skeleton
      ```

  - repository 资源库 （clone的对象）

- 

- 

- 

- **推送本地资源库到GitHub（Push Local Repository to GitHub）**

  ```shell
  git remote add origin https://github.com/w3schools-test/hello-world.git
  ```

  `git remote add origin URL` specifies that you are adding a remote repository, with the specified `URL`, as an `origin` to your local Git repo.

- **push our master branch to the origin url, and set it as the default remote branch**

  ```
  git push --set-upstream origin master
  ```

  

- `pull` is a combination of 2 different commands:

  - `fetch`: gets all the change history of a tracked branch/repo.

    ```shell
    git fetch origin
    ```

    

  - `merge`: combines the current branch, with a specified branch.

  - `pull` is a combination of `fetch` and `merge`. It is used to pull all changes from a remote repository into the branch you are working on.

- Use `pull` to update our local Git: 使用远程库更新本地库

  ```shell
  git pull origin
  ```

  

- `Pull` a GitHub branch to your local Git.

  ```shell
  # 查看本地local和远程remote的所有分支
  git branch -a
  
  # 只查看远程分支
  git branch -r
  
  # 创建同名的本地分支追踪远程分支
  git checkout html-skeleton
  ```

  

- `Push` a Branch to GitHub

  在本地创建新分支

  ```shell
  # 在本地创建新分支
  git checkout -b update-readme
  
  # 添加到缓存区
  git add README.md
  
  # 提交
  git commit -m "Updated readme for GitHub Branches"
  
  # 推送  Push the branch from our local Git repository, to GitHub.
  git push origin update-readme
  ```

  

- Each commit should have a message explaining what has changed and why.

- `clone`: A `clone` is a full copy of a repository, including all logging and versions of files.

  ```shell
  git clone https://github.com/w3schools-test/w3schools-test.github.io.git
  
  # 克隆到指定文件夹
  git clone https://github.com/w3schools-test/w3schools-test.github.io.git myfolder
  
  # 查看资源库日志
  git log
  
  ```

  

- Git can specify which files or parts of your project should be ignored by Git using a `.gitignore` file.

  ```shell
  # Ignore any files with the .log extension
  *.log
  
  # ignore ALL files in ANY directory named temp
  temp/
  ```

  | Pattern                          | Explanation/Matches                                          | Examples                                                     |
  | :------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
  |                                  | Blank lines are ignored                                      |                                                              |
  | # *text comment*                 | Lines starting with # are ignored                            |                                                              |
  | *name*                           | All *name* files, *name* folders, and files and folders in any *name* folder | /name.log /name/file.txt /lib/name.log                       |
  | *name*/                          | Ending with / specifies the pattern is for a folder. Matches all files and folders in any *name* folder | /name/file.txt /name/log/name.log  **no match:** /name.log   |
  | *name*.*file*                    | All files with the *name.file*                               | /name.file /lib/name.file                                    |
  | */name*.*file*                   | Starting with / specifies the pattern matches only files in the root folder | /name.file  **no match:** /lib/name.file                     |
  | *lib/name*.*file*                | Patterns specifiing files in specific folders are always realative to root (even if you do not start with / ) | /lib/name.file  **no match:** name.file /test/lib/name.file  |
  | ***/lib/name.file*               | Starting with ** before / specifies that it matches any folder in the repository. Not just on root. | /lib/name.file /test/lib/name.file                           |
  | ***/name*                        | All *name* folders, and files and folders in any *name* folder | /name/log.file /lib/name/log.file /name/lib/log.file         |
  | /lib/***/name*                   | All *name* folders, and files and folders in any *name* folder within the lib folder. | /lib/name/log.file /lib/test/name/log.file /lib/test/ver1/name/log.file  **no match:** /name/log.file |
  | *.*file*                         | All files withe *.file* extention                            | /name.file /lib/name.file                                    |
  | **name*/                         | All folders ending with *name*                               | /lastname/log.file /firstname/log.file                       |
  | *name*?.*file*                   | ? matches a **single** non-specific character                | /names.file /name1.file  **no match:** /names1.file          |
  | *name*[a-z].*file*               | [*range*] matches a **single** character in the specified range (in this case a character in the range of a-z, and also be numberic.) | /names.file /nameb.file  **no match:** /name1.file           |
  | *name*[abc].*file*               | [*set*] matches a **single** character in the specified set of characters (in this case either a, b, or c) | /namea.file /nameb.file  **no match:** /names.file           |
  | *name*[!abc].*file*              | [!*set*] matches a **single** character, **except** the ones spesified in the set of characters (in this case a, b, or c) | /names.file /namex.file  **no match:** /namesb.file          |
  | *.*file*                         | All files withe *.file* extention                            | /name.file /lib/name.file                                    |
  | *name*/ !*name*/secret.log       | ! specifies a negation or exception. Matches all files and folders in any *name* folder, except name/secret.log | /name/file.txt /name/log/name.log  **no match:** /name/secret.log |
  | *.*file *!*name*.file            | ! specifies a negation or exception. All files withe *.file* extention, except name.file | /log.file /lastname.file  **no match:** /name.file           |
  | *.*file *!*name*/**.file* junk.* | Adding new patterns after a negation will re-ignore a previous negated file All files withe *.file* extention, except the ones in *name* folder. Unless the file name is junk | /log.file /name/log.file  **no match:** /name/junk.file      |

​	   To create a `.gitignore` file, go to the root of your local Git, and create it:

​		



- 