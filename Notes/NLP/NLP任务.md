#### 文本语义匹配任务

- 数据格式：分为`pairwise`和`pointwise`两种模式。

  - pairwise模式

    训练集（train）格式如下： query \t pos_query \t neg_query。 query、pos_query和neg_query都是以空格分词的中文文本，三个文本之间中间使用制表符'\t'隔开，pos_query表示与query相似的正例，neg_query表示与query不相似的随机负例。

    ```sql
    现在 安卓模拟器 哪个 好 用     电脑 安卓模拟器 哪个 更好      电信 手机 可以 用 腾讯 大王 卡 吗 ?
    土豆 一亩地 能 收 多少 斤      一亩 地土豆 产 多少 斤        一亩 地 用 多少 斤 土豆 种子
    ```

    开发集（dev）和测试集（test）格式：query1 \t query2 \t label。

    query1和query2表示以空格分词的中文文本，label为0或1，1表示query1与query2相似，0表示query1与query2不相似，query1、query2和label中间以制表符'\t'隔开。

    ```sql
    现在 安卓模拟器 哪个 好 用    电脑 安卓模拟器 哪个 更好      1
    为什么 头发 掉 得 很厉害      我 头发 为什么 掉 得 厉害    1
    常喝 薏米 水 有 副 作用 吗    女生 可以 长期 喝 薏米 水养生 么    0
    长 的 清新 是 什么 意思     小 清新 的 意思 是 什么 0
    ```

  - pointwise模式：

    训练集、开发集和测试集数据格式相同：query1和query2表示以空格分词的中文文本，label为0或1，1表示query1与query2相似，0表示query1与query2不相似，query1、query2和label中间以制表符'\t'隔开。

    infer 数据集：`pairwise`和`pointwise`的infer数据集格式相同：query1 \t query2。

- 模型结构

  ![image-20230311230924232](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230311230924232.png)

  ![image-20230313111252212](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230313111252212.png)

  - 单塔模型：便于直接进行分类

    ![image-20230311230946913](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230311230946913.png)

  - 双塔模型：便于获取词向量
  
    ![image-20230311230938156](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230311230938156.png)

#### 情感分析任务

![image-20230312102614659](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230312102614659.png)

- 情感分析划分

  - 词级情感分析

    ![image-20230312103458368](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230312103458368.png)

    离散表示方法无法表示情感的强弱，无法衡量情感词之间的差异，连续多维表示方法则是将情感词表示为多维向量，（情感极性的正负、情感强弱、情感的主客观信息）

  - 句子级情感分析/篇章级情感分析

  - 属性级情感分析

    ![image-20230312105335936](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230312105335936.png)

    

- 情感分析应用

  - 电商评论情感分析

#### 问答系统

- 开放域问答：不限定领域，从大规模语料库中查找答案。

- 检索式问答：

  - DrQA论文
    - 段落检索（Retriever）：从大规模语料库中检索出相关文档或段落
    - 阅读理解（Reader）：从候选文档或段落中获取答案。
  - 评估方式
    - EM（exact match）完全匹配
    - F1 Score：部分匹配，计算字面重合程度
  - 经典数据集：
    - SQuAD，第一个大规模有监督数据集
    - DuReader：中文阅读理解数据集
  - 阅读理解模型存在的问题
    - 过稳定：对于不同的提问，给出相同的答案
    - 过敏感：对于相同语义的提问，给出不同的答案。

  - 检索方式
    - 稀疏向量检索：字符匹配类型，字面匹配，倒排索引
      - TF-IDF和BM25表示形式
    - 稠密向量检索：语义匹配类型
      - 负采样方式构建训练数据集，改进训练方式提升稠密向量检索效果的方法（DPR Dense Passage Retrieval）
  - 实现步骤：
    - 从大规模文档中检索得到Top-K个段落
      - 正例
      - 负例
        - 强负例，有点话题相关，但是不是问题的答案，比较重要
        - 弱负例，随机选取的段落，可能是和问题完全无关的。
    - 从K个候选段落中抽取答案：阅读理解任务

#### 模型压缩

- 模型压缩基本方法
  - 量化
  - 裁剪
    - 只能进行压缩，无法实现推理加速
  - 蒸馏
    - 需要重新进行训练
    - 不改变预测方法，压缩的同时可以进行推理加速