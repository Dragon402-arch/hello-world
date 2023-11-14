#### Chatbot（聊天机器人）

- 聊天机器人可分为：

  -  Task-Oriented Chatbots：The task-oriented chatbots are designed to perform specific tasks. 

  - General Purpose Chatbots：general purpose chatbots can have open-ended discussions with the users.
  - Hybrid chatbots that can engage in both task-oriented and open-ended discussion with the users.
- Approaches for Chatbot Development：

  - Learning-Based Chatbots
    - Retrieval-based chatbots : **learn to select a certain response to user queries.**
    - Generative chatbots：**learn to generate a response on the fly.**
  - Rule-Based Chatbots
    - There are a specific set of rules. If the user query matches any rule, the answer to the query is generated, otherwise the user is notified that the answer to user query doesn't exist.
- 检索类聊天机器人：回答常见问题（搜索）
  - 过程：
    - 识别请求（request）：意图识别
    - 返回响应（response）
  - 实例：电商自动回复
  - 情感分类：积极、中性、消极
    - 若识别为消极，可以提供额外帮助，如转人工客服
    - [讲解](https://poshai.medium.com/ai-academy-introduction-to-natural-language-processing-e4b8f9bf6396)
- 生成类聊天机器人
  - 对话
  - 总结损失函数维度
  - text2sql
  - batch normalization and layer normalization
  - token level task  and sentence level task

### 对话系统

- 对话任务
  - 任务型：帮助用户完成特定任务
    - 智能家居：打开空调
    - 消费电子：电话手表
    - 车载出行：完成指令
    - 智能客服
  - 闲聊型：开放域对话系统
    - 检索型系统：从语料库中选取一个合适的句子作为回复
      - 返回最相似的句子对应的回复
      - 直接返回最相似的句子
    - 生成式系统：使用模型根据上一句生成回复。
      - 生成模型
      - 解码策略
        - Greedy decoding
        - Beam Search
        - Random Sampling
        - Top k Sampling
        - Top p Sampling
      - 对话评估
        - 自动评估
          - BLEU
          - distinct
        - 人工评估
  - 问答型：知识问答

#### NLP任务

- 词向量应用与展示

- 文本语义相似度计算：文本语义匹配任务，[代码](https://aistudio.baidu.com/aistudio/projectdetail/2029701)

- 使用预训练模型实现快递单信息抽取：实体抽取任务，[代码](https://aistudio.baidu.com/aistudio/projectdetail/2065855)

- 基于预训练模型完成实体关系抽取：关系抽取任务，

- 文本情感分析

- 机器阅读理解

  - 检索式问答（两阶段）
    - 段落检索（Retriever）：从大规模语料库中检索出相关文档或段落
      - 检索方式
        - 稀疏向量检索：字符匹配类型，字面匹配，倒排索引
          - TF-IDF和BM25表示形式
        - 稠密向量检索：语义匹配类型
      - 实现步骤：
        - 从大规模文档中检索得到Top-K个段落
          - 正例
          - 负例
            - 强负例，有点话题相关，但是不是问题的答案
            - 弱负例，随机选取的段落，可能是和问题完全无关的。
        - 从K个候选段落中抽取答案：阅读理解任务
    - 阅读理解（Reader）：从候选文档或段落中获取答案。
  - 划分
    - 抽取式阅读理解
      - 任务定义：
        - 输入：给定篇章P和问题Q
        - 输出：答案开始位置与结束位置
    - 生成式阅读理解

- 中英文本翻译系统

- 动手搭建轻量级机器同传翻译系统

- 对话意图识别

- 动手搭建中文闲聊机器人

- 预训练模型小型化与部署实战

  ![image-20230310171503998](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230310171503998.png)
