| **方法**                                     | **作用**                                                     |
| -------------------------------------------- | ------------------------------------------------------------ |
| **os.getcwd****()**                          | 获取当前工作目录，即当前程序文件所在的文件夹                 |
| **os.chdir****(****path****)**               | 改变当前工作目录，**需传递新的路径**                         |
| **os.listdir****(****path****)**             | 返回指定路径下的**文件名称列表**                             |
| **os.mkdir****(path)**                       | **在某个路径下创建文件夹，找不到相应的路径时报错**           |
| **os.makedirs****(path)**                    | 递归创建文件夹，**找不到路径时自动创建**                     |
| **os.removedirs****(path)**                  | **递归删除文件夹**，必须都是空目录，如果不是空文件夹将会报错 |
| **os.rename****(****src****,** **dest****)** | **文件或文件夹重命名**                                       |
| **os.path.split****(path)**                  | **将文件路径****path****分割成文件夹和文件名，并将其作为二元组返回** |
| **os.path.abspath****(path)**                | 返回path规范化的**绝对路径**                                 |
| **os.path.join****(path1, path2[, ...])**    | **将多个路径组合**后返回，例如将文件夹和里面的文件组合得到绝对路径 |
| **os.path.getsize****(path):**               | **返回文件大小，以字节为单位**                               |

- 文本聚类的用途是：

  - 文本归类
  - 寻找主题

- 聚类划分：

  In **hard** **clustering**, every object belongs to *exactly one* cluster. In **soft** **clustering**, an object can belong to *one or more* clusters.

  - 软聚类

  - 硬聚类

    ![image-20220105102921856](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20220105102921856.png)

