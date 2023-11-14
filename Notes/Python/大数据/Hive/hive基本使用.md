- 查看库与表

  ```sql
  # 查看库
  show databases;
  
  # 查看表
  show tables;
  ```

- 查看表结构

  ```sql
  desc r2wb2r;
  ```

- 查看列名：

  ```shell
  show columns in person;
  
  [('sg',), ('xm',), ('csrq',), ('hyzk',), ('hjd',), ('mz',), ('xb',), ('gmsfhm',), ('xl',), ('rybq',)]
  (166, '班嘉城', '1996-06-14', '离异', '黑龙省江七台河市新兴区', '苗族', '女', '230902199606143520', '博士研究生', '流动人口,前科人员,交通违章')
  
  ```

   

- 导入多条数据

  ```sql
  INSERT INTO  table person (sg,xm,csrq,hyzk,hjd,mz,rybq,xb,gmsfhm,xl) VALUES (178, "汤榕", "1995-11-06", "丧偶", "浙江省杭州市富阳市", "汉族", "常住人口,涉赌", "男", "330183199511067439", "博士研究生"),(171, "张振宇", "1995-12-03", "丧偶", "浙江省杭州市富阳市", "汉族", "常住人口,涉赌", "男", "330183199511067439", "博士研究生")
  
  
  
  
  
  
  
  
   CREATE TABLE tetris.person (                             
     sg int COMMENT '身高',                          
     xm string COMMENT '姓名',                      
     csrq string COMMENT '出生日期',                   
     hyzk string COMMENT '婚姻状况',                   
     hjd string COMMENT '户籍地',                    
     mz string COMMENT '民族',                     
     xb string COMMENT '性别',                        
     gmsfhm string COMMENT '公民身份号码',               
     xl string COMMENT '学历',                        
     rybq string COMMENT '人员标签')  
  
  
  ```

- 

