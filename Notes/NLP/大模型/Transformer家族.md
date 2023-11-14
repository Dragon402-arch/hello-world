### Transformer 家族

- Transformer

  - Encoder:BERT

    - 用于自然语言理解任务


  - Decoder: GPT-3

    - 用于自然语言生成任务
    - ChatGPT：
      - 靠GPT 3.5本身，尽管它很强，但是它很难理解人类不同类型指令中蕴含的不同意图，也很难判断生成内容是否是高质量的结果。


  - Encoder-Decoder:T5(the **T**ext-**T**o-**T**ext **T**ransfer **T**ransformer)、BART、UniLM（Prefix LM），NEZHA

    - 条件性文本生成任务
    - StructBERT、BART模型：适用于语法纠错任务的模型

  - 博客讲述：
    - [NLP系列之预训练模型（一）](https://zhuanlan.zhihu.com/p/351504576)
    - [NLP系列之预训练模型（二）](https://zhuanlan.zhihu.com/p/355366424)
    - [NLP系列之预训练模型（三）](https://zhuanlan.zhihu.com/p/356867198)
    - [NLP系列之预训练模型（四）](https://zhuanlan.zhihu.com/p/376499743)
    - [NLP系列之预训练模型（五）](https://zhuanlan.zhihu.com/p/382678571?utm_id=0)
    - [NLP系列之预训练模型（六）](https://zhuanlan.zhihu.com/p/406743805)

#### BERT

- 显卡：NVIDIA Tesla A100，显存 40G：68999；80G：95999

- **Masked Language Model （MLM）**

  - In MLM, the model masks some of the values from the input and tries to predict the masked (missing) word based on its context. 
  - MLM任务会引起一个问题：**预训练和下游精调任务输入不一致，因为下游任务的时候，输入是不带【MASK】的，这种不一致会损害BERT的性能**，这也是后面研究的改善方向之一。BERT自身也做出了一点缓解，就是对这15%要预测的token的输入进行一些调整，对输入的句子进行WordPiece处理后，选取 15 % 的 token 进行 mask，其中
    - 80%用【MASK】代替；
    - 10%用随机的词token替；
    - 10%保持原来的token不变

- **Next Sentence Prediction (NSP)**

  - During training the model gets input pairs of sentences and it learns to predict if the second sentence is the next sentence in the original text as well.

- **WordPiece**:**引入WordPiece作为输入，可以有效缓解OOV（Out Of Vocabulary）问题，且进一步增加词表的丰富度和表征能力**

  - character level的Byte-Pair Encoding（BPE）编码

    - 不用word级别（有些word是unseen，虽然可以用UNK代替，但模型对这种word无法建模），也不用character级别（unseen这个单词中的un，这个word中如果分别对u、n进行建模，则模型无法识别到合起来的否定意思），于是选择subword（parts of word）最好。

      ![image-20230220221151851](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230220221151851.png)

      BPE 是一个进行 subword tokenization 的常用方法，使用该方法得到的vocabulary 包含了最常见的word和subword。

    - Natural Language Inference：判断输入的两个句子是A可以推出B的关系、A和B是矛盾的关系、还是A和B没有关系。

    - Word
    
      - Vocabulary from training data 
    
      - Issue：unseen words cannot be well modeled，but human can
    
    - Character
      - 优点：no unseen，small vocab
      - 缺点：semantics of multiple  characters is difficult to model
    - Subword（parts of word）
      - A balance between word and character
      - Subword modeling address issues about unseen words
      - Usually include frequent words and frequent subwords.

- 模型评价：

  - 优点：双向上下文编码，表征能力强
  - 缺点：预训练和精调任务不一致。


#### **RoBERTa **

- **（Robustly Optimized BERT pre-training Approach）**

- **Dynamic Masking**: BERT uses *static masking* i.e. the same part of the sentence is masked in each Epoch. In contrast, RoBERTa uses *dynamic masking,* wherein for different Epochs different part of the sentences are masked. This makes the model more robust.

  BERT模型使用的是静态掩码（训练时每个Epoch给定句子中被掩盖的部分是固定的）

  RoBERTa则使用的是动态掩码（训练时每个Epoch给定句子中被掩盖的部分是随机的）

- **Remove NSP Task**：NSP任务不是很有用

- **More data Points**：BERT的训练数据量为 16 GB，而 RoBERTa 的训练数据量为 160 GB.

  - RoBERTa：train only with full-length sequences
  - BERT：train on the reduced length sequences

- **Large Batch size**: RoBERTa used a batch size of 8,000 with 300,000 steps. In comparison, BERT uses a batch size of 256 with 1 000,000 steps.

- 基于bytes-level的Byte-Pair Encoding （BPE）编码

总而言之，在 BERT 模型的基础上,去除 NSP 任 务,使用更大规模的数据集和更大的 batch-size 再次训练得到的预训练语言模型。

#### **RoBERTa-WWM （Whole World Mask）**

- **Whole World Mask**：为适应中文的语言特点，在模型训练时采用全词遮挡的方式，提升 了 RoBERTa 模型在中文环境下的文本表示能力。
  - 示例：
    - 原句：使用语言模型来预测下一个词出现的概率
    - 遮挡：使用语言*模型*来*预测*下一个词出现的*概率*
    - 经典遮挡：使用语言[MASK]型来[MASK]测下一个词出现的[MASK]率
    - 新版遮挡：使用语言[MASK]来[MASK]下一个词出现的[MASK]

#### **ALBERT**

-  **Cross-layer parameter sharing**
- **Factorized embedding layer parameterization**
-  SOP (Sentence Order Prediction)：是否为正确的顺序，二分类任务。

#### ERNIE

- 百度ERNIE：百度ERNIE的思路是：在预训练阶段被Mask掉的对象上做文章，我们可以使用比如命名实体识别工具／短语识别工具，将输入中的命名实体或者部分短语Mask掉，这些被Mask掉的片断，代表了某种类型的语言学知识，通过这种方式，强迫预训练模型去强化地学习相关知识。

  ![image-20230220231318604](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230220231318604.png)

- Knowledge Integration
  - MASK 类型
    - Basic-level Masking：和BERT一样
    - Entity-level Masking：把实体作为一个整体【MASK】
    - Phrase-Level Masking：把短语作为一个整体【MASK】
- Dialogue Language Model（DLM）
  - BERT预训练任务NSP 被用该任务代替了，训练使用的是多轮问答的数据，不仅需要预测被MASK掉的token，还要预测输入的内容是否为真实的问答对。

#### ERNIE 2.0

- continual multi-task learning 连续多任务学习：让模型学习多个任务，进行联合训练，有三种策略：

  ![image-20230303105502006](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230303105502006.png)

  - Multi-task Learning：让模型同时学这3个任务，并将这3个任务的损失值添加权重求和，然后一起反向传播更新参数；
  - Continual Learning：先训练任务1，再训练任务2，再训练任务3，这种策略的缺点是容易遗忘前面任务的训练结果，最后训练出的模型容易对最后一个任务过拟合；
  - Sequential Multi-task Learning：连续多任务学习，即第一轮的时候，先训练任务1，但不完全让它收敛训练完，第二轮，一起训练任务1和任务2，同样不让模型收敛完，第三轮，一起训练三个任务，直到模型收敛完。

- 预训练任务

  ![image-20230303111249300](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230303111249300.png)

  - word-aware tasks：learn the lexical representations，学习词汇表示
    - Knowledge Masking 
    - Token-Document Relation：预测在段落A中出现的token，是否在文档的段落B中出现。
    - Capital Prediction 大写单词预测任务，在句子中，与其他单词相比，大写单词通常具有某些特定的语义信息。
      - add a task to predict whether the word is capitalized or not.预测一个单词是不是大写单词。
  - structure-aware tasks ：learn the syntactic representations，学习句法表示
    - Sentence Reordering Task ：把文档中的句子打乱，预测正确顺序；
    - Sentence Distance Task：分类句子间的距离（0：表示两个句子在同一篇文档内且是相邻的，1：表示两个句子在同一篇文档内但不是是相邻的，2：表示两个句子不在同一篇文档内）。
  - semantic-aware tasks：learn the semantic representations，学习语义表示
    - Discourse Relation Task:计算两个句子之间的语义和修辞关系；
    - IR Relevance Task： It is a 3-class classification task which predicts the relationship between a query and a title. query作为第一个句子，而title作为第二个句子。短文本信息检索关系（0：搜索并点击，1：搜素并展现未点击，2：无关）。

#### SpanBERT

- **“Span of words” masking than a “Random masking”**

  - **Tokens =** [Data, science**,** combines, domain, expertise, programming, skills, and, knowledge, of, mathematics, and, statistics]
  - In MLM case, random masking is performed:[Data, science, [MASK], domain, [MASK], programming, skills, [MASK], knowledge, of, mathematics, [MASK], statistics]
  - However, for span, the masking is randomly done but on a span of continuous words:[Data, science, combines, [MASK], [MASK], [MASK], [MASK], [MASK], knowledge, of, mathematics, and, statistics]

- **Overall loss function of SpanBERT = MLM objective Loss + SBO Objective Loss**

  -  **MLM Objective**: Predict the masked token i.e. probability of all the words in the vocabulary of being the masked word.

    预测被遮挡的 token，即 词汇表中所有词可能作为被遮挡 token 的概率

  - **Span Boundary Objective (SBO):** For SBO, instead of taking the representation of the masked token, the **representation of the adjacent tokens** are taken into account. In our example: [Data, science, **combines**, [MASK], [MASK], [MASK], [MASK], [MASK], **knowledge,** of, mathematics, and, statistics]. Representation of the words **“combines” and “knowledge”** will be used to derive the Masked tokens. 不是使用被遮挡的 tokens 来预测 span 的边界，而是使用边界两端相邻的 token。

    ![image-20230220232601648](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230220232601648.png)

- 主要用于问答和关系抽取任务。

  ![image-20230220233528055](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230220233528055.png)

#### DistillBERT and TinyBERT

- knowledge distillation（知识蒸馏）：try to reduce the size of the model by creating a smaller model which somehow replicates the output of the bigger model.
- distillBERT 是从教师模型的输出层蒸馏知识，比如 原BERT模型是12层，那么 distillBERT 模型可能只用6层去学习教师模型的输出；而TinyBERT模型是不仅从教师模型的输出层蒸馏知识，而且从教师模型的中间层也蒸馏知识。
- **TinyBERT = DistilBERT + distilled knowledge from the intermediary layers**

#### **ELECTRA **

- **Efficient Learning an Encoder that Classifies Token Replacements Accurately**

- **Replaced Token Detection（RTD）**：In RTD, instead of masking the token, the token is replaced by a wrong token and the model is expected to classify, whether the tokens are replaced with wrong or not.（**Replaced token detection task**）

  将文本中正常的 token 用 别的 token 进行替换，训练模型识别每个 token 是否为替换过来的。 

- **Remove NSP Task**

- **Better Training:** Instead of training on the mask data only, the ELECTRA model the training is done on the complete data.

  模型是在完整数据上进行训练的，而BERT则是在掩码后的数据上进行训练的。

#### Transformer-XL(extra-long）

- [讲解](https://towardsdatascience.com/transformer-xl-review-beyond-fixed-length-contexts-d4fe1d6d3c0e)

- Its key innovations are a **segment-level recurrence mechanism** and a novel **positional encoding scheme**.

- segment-level recurrence：

  - Previous segment  embeddings are fixed and cached  to be reused when training the next segment

  - increases the largest dependency length  by N times （N：network depth ）

    - 下面的示例中深度为3，而 length 为4，理论上能建模12个token的依赖关系，但是由于有两个token重合了，因此实际上只建模了10（length * depth -（depth-1）=（length -1）* depth + 1）个长度的依赖关系

  - 计算举例：先计算第一个片段 [1,2,3,4]，并把数据缓存下来，在计算[5,6,7,8]这个片段的 5 会把[2,3,4,5]的数据带入进行计算。

    ![ ](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230221204856554.png)

- Relative Position Encoding:[0,1,2,3,0,1,2,3]原本的transformer 使用的是绝对位置编码，而在循环时每个片段的绝对位置会有重合的情况，需要换成相对位置编码。

- Unlike the traditional Transformer model, it can capture longer-term dependency and solve the context fragmentation problem, which are the main limitations of the vanilla Transformer.

  也就是传统的 Transformer 无法建模长期依赖关系（最长512）并且 上下文被碎片化的问题。

#### XLNet

![image-20230221232243692](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230221232243692.png)

![img](https://miro.medium.com/max/770/1*kjWzIUND0HsQvGNmMj2Pjw.png)

#### MacBERT

- MLM as correction BERT（纠错型掩码语言模型预训练任务）：为了缓解**BERT模型在预训练和精调阶段的不一致**，对MLM 任务进行了改进。

- 改进策略

  - whole word masking (wwm) and N-gram masking ：**虽然【MASK】是对分词后的结果进行，但在输入的时候还是单个的token。**对于分词后的结果进行 n-gram 进行 mask 时，对应的 n-gram 被 mask 的概率如下：
    - unigram：40%
    - bigram：30%
    - trigram：20%
    - 4-gram：10%
  - 近义词替换掉[MASK]符号：掩码语言模型（MLM）中，引入了[MASK]标记进行掩码，但[MASK]标记并不会出现在下游任务中。在MacBERT中，**我们使用相似词来取代[MASK]标记**。相似词（近义词）通过Synonyms toolkit 工具获取，算法基于word2vec进行相似度计算。当要对N-gram进行掩码时，我们会对N-gram里的每个词分别查找相似词。当没有相似词可替换时，我们将使用**随机词**进行替换。
  - 对基于分词后的结果随机挑选15%的词进行【MASK】，其中
    - 80%用同义词代替；
    - 10%用随即词代替；
    - 10%保持不变。
  - 弃用 NSP 任务，改换为和ALBERT使用的 SOP 任务。

  ![image-20230303153322200](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230303153322200.png)

#### StructBERT

- 从**单词**和**句子**的两个角度来提升BERT对语言结构的理解。（阿里达摩院提出的模型）

- 增加了两个预训练任务

  - Word Structural Objective：一个好的语言模型必须能够通过句子中打乱顺序的单词组**恢复**出单词的原来顺序。通过对子序列（子序列的长度定为3比较好，不大不小）打乱token 顺序，训练模型恢复其原本的顺序，形成一个单词级别的任务。该任务主要针对单句子任务。
  - Sentence Structural Objective：针对原本的 NSP 任务进行改进，判断第二个句子与第一个句子的关系：
    -  the next sentence（下一句）
    - the previous sentence（前一句） 
    - a random sentence（其他文档的随机一句）

  ![image-20230303160456214](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230303160456214.png)

  

#### GPT 

- GPT-3:Although GPT-3 introduced remarkable advancements in natural language processing, it is limited in its ability to align with user intentions.模型的输出（回答）会出现和用户的意图答非所问的情况。
  - 回答的结果对用户没有帮助，这意味着它不遵循用户的明确说明。
  - 包含反映不存在或不正确事实
  - 缺乏可解释性，使人类很难理解模型如何做出特定的决策或预测
  - 包括有害或冒犯性并传播错误信息的有毒或有偏见的内容。

- ChatGPT :introduced a novel approach to incorporating human feedback into the training process to better align the model outputs with user intent.
  - **Reinforcement Learning from Human Feedback (RLHF)**

- 

- 

- 

- 

- 

- 

- 

- 

- 

- 

- 

- 实验方法

  - 消融实验(ablation experiment)（控制变量法）

    - 你论文提了三个贡献点：A、B、C。

      你去掉A，其它保持不变，发现效果降低了，那说明A确实有用。

      你去掉B，其它保持不变，发现效果降的比A还多，说明B更重要。

      你去掉C，其它保持不变，发现效果没变，那C就是凑字数的。

  - 对比实验

    - 方案A与方案B、C、D的效果进行比较。

