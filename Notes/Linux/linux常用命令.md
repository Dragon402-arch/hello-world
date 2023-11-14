#                                       **linux常用命令**

### 1、SCP服务器文件传输：

#### 	1.1、linux服务器之间文件传输

- SCP命令获取远程服务器上的目录：

  ```shell
  scp -P port -r user@ip:/home/xxx/ /home/xxx/
  ```

- SCP命令获取远程服务器上的文件：

  ```sh
  scp -P port user@ip:/home/xxx.xxx /home/xxx.xxx
  # 示例（前者是远程路径，后者是本地路径）
  
  scp -P 22 -r 192.168.51.59:/home/chinaoly/lis/eventTrain/ /home/chinaoly/anaconda3/envs/intentsearch.gz 
  
  scp -P 22 -r 192.168.51.58:/home/chinaoly/lis/entityExtraction/entity_data /home/chinaoly/lis/alg_dataset/
  
  
  scp -P 22 -r chinaoly@192.168.51.58:/home/chinaoly/lis/db_data.tar.gz /home/lee/esTest/es_test/
  
  
  scp -P 22 192.168.51.58:/home/chinaoly/lis/dependent_files/intent_env.gz /home/zkb/miniconda3/envs
  
  
  scp -P 22 -r 192.168.51.58:/home/chinaoly/lis/dependent_files/intent_env.gz /home/zkb/miniconda3/envs
  Chinaoly@159753
  
  ```
  
- SCP命令将本地目录上传到远程服务器：

  ```shell
  scp -P port  -r /home/xxx/ user@ip:/home/xxx/
  ```

- SCP命令将本地文件上传到远程服务器：

  ```shell
  scp -P port  /home/xxx.xxx user@ip:/home/xxx.xxx
  # 示例（前者是本地路径，后者是远程路径）
  scp -P 22 -r /home/chinaoly/lis/textCls/table_cls chinaoly@192.168.51.59:/home/chinaoly/lis/tableClassification/
  
  sudo scp -P 22 /home/chinaoly/lis/eventTrain/SPN4EE_V1/final_trained_event_model.pth chinaoly@192.168.51.51:/home/chinaoly/lis
  
  
  sudo scp -P 22 README.md root@172.18.155.7: /root/
  ```

#### 1.2、Windowst与Linux服务器之间互传（以下命令均为在Windows命令行中输入）：

- Windows本地传输文件到指定Linux服务器

  ```shell
  # 上传一个文件到Linux服务器
  scp D:\demo.txt  chinaoly@192.168.51.58:/home/chinaoly/lis/
  
  scp E:\Pycharm\torch_env.gz chinaoly@192.168.51.58:/home/chinaoly/lis/arm-centos7.5-torch-env/
  
  scp E:\Python-3.7.10.tgz chinaoly@192.168.51.58:/home/chinaoly/lis/arm-centos7.5-torch-env/
  
  scp E:\case_event_data.json chinaoly@192.168.51.58:/home/chinaoly/lis/arm-centos7.5-torch-env/
  
  scp E:\Pycharm\intentSearch\intent_search_test\target_file\intent_test_model.pth lee@124.70.134.105:/home/lee/pythonProject/pretrained_models/
  
  scp E:\case_train_data.json chinaoly@192.168.51.58:/home/chinaoly/lis/eventExtraction/
  # 上传一个文件夹时要在scp命令中加参数 -r
  scp -r E:/intentsearch chinaoly@192.168.51.58:/home/chinaoly/rxy/IntentSearch
  
  ```

- Linux服务器传输文件到Windows

  ```shell
  # 从Linux服务器上传输文件到Windows(CMD输入命令)
  scp -P 22 root@119.3.170.185:/myProject/torch_env/bin/torch_env.gz E:\Pycharm
  
  
  scp -P 22 chinaoly@192.168.51.51:/home/chinaoly/lis/operator_recommend1.1.0.tar.gz E:\
  
  scp -P 22 lee@124.70.134.105:/home/lee/pythonProject/search_venv_1_1.tar.gz E:\
  
  scp -P 22 chinaoly@192.168.51.51:/home/chinaoly/anaconda3/envs/intentsearch.gz E:\Pycharm
  
  scp -P 22 chinaoly@192.168.51.58:/home/chinaoly/lis/cuda-repo-rhel7-10-0-local-10.0.130-410.48-1.0-1.x86_64 E:\
  ```



### 2、linux文件复制与删除

#### 		1、复制

- 将一个文件夹下的所有内容复制到另一个文件夹下：

  ```shell
  cp -r /home/chinaoly/KG_platfrom/NER_knowledge_v11/. /home/chinaoly/lis/updateProject/
  
  cp -r /home/chinaoly/lis/createData/generate_name/person_infos /data1/intent_search_data/
  /data1/intent_search_data/
  cp -r /home/chinaoly/lis/createData/generate_hotel_data/hotel_relation_data /data1/intent_search_data/
  
  cp -r /data1/bert/data/jcy/rltrxd/jcy_rltrxd200/. /home/chinaoly/lis/entityExtraction 
  
  ```

- 复制一个文件到另一个目录下：

  ```shell
  cp -r text.txt /home/chinaoly/rxy/IntentSearch/intentsearch/intent_recognition
  
  cp  /data1/ChinaolyDataSet/领域数据集/其他/浙江警情数据/dq100_tmp.zip  /home/chinaoly/lis/textCluster
  
  ```

#### 		2、删除

- 删除某个文件夹下的所有文件，但不删除文件夹：

  ```shell
  # 第一步
  cd target_file 
  
  # 第二步  -rf为参数-r和-f 的简写  -r:递归删除  -f:强制删除
  rm -rf *         
  ```

- 删除某个空文件夹：

  ```shell
  rmdir 文件夹名
  ```

- 直接删除文件夹及其下面的所有文件：

  ```shell
  rm -rf 文件夹名
  
  
  148.144.30.4 hadoop.hadoop.com.
  ```

- 删除某个文件

  ```linux
  rm -f 文件名
  ```


- 删除当前目录下所有以test开头的目录和文件

  ```
  rm -r test*
  ```

  

## 3、其他命令

- 清屏命令

  ```shell
  # linux系统下清屏：
  clear
  # window系统下清屏：
  cls
  ```

- 创建虚拟环境的命令

  ```shell
  # 创建虚拟环境的命令
  conda create -n pytorch python=3.7
  
  # 激活环境
  conda activate bert
  
  # 查看conda下的虚拟环境
  conda env list
  
  # 激活miniconda3中虚拟环境使用的语句
  # $ export PATH=/home/zkb/miniconda3/envs/intentsearch/bin/:$PATH
  
  $ export PATH=/home/chinaoly/miniconda3/bin:$PATH
  $ source ~/.bashrc
  ```

- 后台训练

  ```shell
  nohup python -u intentsearch_router.py > intentsearch_router.py.log 2>&1 &
  
  
  nohup python -u Knowledgebase_router.py  > run.log 2>&1 &
  nohup python -u router.py  > router.py.log 2>&1 &
  nohup python -W ignore -u run.py --gpu_devices 0 1 2 3 > run.log 2>&1 &
  
  nohup python -u run.py > run.py.log 2>&1 &
  # 查看后几行
  tail -f train.log
  
  
  
  nohup python -u import_data_to_es.py > run.log 2>&1 &
  ```
  
- 其他情况 

  ```shell
  # 查看后台gpu使用情况：
   watch -n 0 nvidia-smi
  # 查看cuda版本 
   nvcc -V
   
   Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim
   
   
  没有报错，但是模型评价指标在第三个epoch之后就不再变化了，而模型的损失值是在变化的，说明模型此时的学习率变为0了，再怎么训练权重都不会再更新了。
  ```

- pytorch与tensorflow

  ```python
  # 查看TensorFlow中的gpu是否可用：
  tf.test.is_gpu_available()
  # 查看版本
  tf.__version__
  # 查看cuda是否可用：
  torch.cuda.is_available()
  # 查看可用的GPU数量：
  torch.cuda.device_count()
  ```

- 查看环境变量

  ```shell
  echo $PATH
  ```

  

- 复制与粘贴

  - 复制：选中即可。
  - 粘贴：shift + insert  

- 模糊查找文件

  ```shell
  # 查找文件名中带有importer
  locate importer
  ```

  



# 4、hugegraph启动

##### 59服务器hugegraph使用

- sever启动路径：

  ```shell
  # 切换当前目录
  cd /home/chinaoly/huge/hugegraph/bin
  
  # 执行该语句，启动sever
  hugegraph-server.sh
  ```

- studio启动路径：

  ```shell
  # 切换当前目录
  cd /home/chinaoly/huge/hugegraph-studio-0.9.0/bin/
  
  # 执行该语句，启动studio
  hugegraph-studio.sh 
  ```

