#### 1、Bag of Words

- 分类
  - Single word
  - N-gram

- 词袋模型：The bag of words approach works fine for converting text to numbers. However, it has one drawback. **It assigns a score to a word based on its occurrence in a particular document.** It doesn't take into account the fact that **the word might also be having a high frequency of occurrence in other documents as well**. Words that occur in almost every document are usually not suitable for classification because they do not provide any unique information about the document.

  只考虑了一个document中的高频词，而没有考虑到该高频词在其他document中也是高频词，此时该高频词对每篇document都没有提供有价值的信息，因此是不建议作为特征的。通过对词频设定阈值如1500，可以限定向量的长度为1500。

- 缺点：

  - 稀疏矩阵，浪费内存空间，

  - 文本表示使用高维特征向量。虽然可以使用词频限制来降低向量的维度，但是仍然会很高。

  - 没有考虑词序

    A type of bag of words approach, known as n-grams, can help maintain the relationship between words. **N-gram refers to a contiguous sequence of n words. ** Although the n-grams approach is capable of capturing relationships between words, the size of the feature set grows exponentially with too many n-grams.

    N-gram 方法虽然可以保存词序关系，但是特征集的大小会因为N-gram的个数而指数级增加。

    - **N-Grams as Features**

      **In the bag of words and TF-IDF approach, words are treated individually and every single word is converted into its numeric counterpart. The context information of the word is not retained.** The N-Grams model basically helps us capture the context information.The intuition behind the N-gram approach is that words occurring together provide more information rather than those occurring individually. Consider for example the following sentence:

    > sentence = "Manchester united is the most successful English football club"
    >
    > 对于这个sentence使用单个word，可以创建如下特征集：
    >
    > ​		Features = {Manchester, United, is, the, most, successful, English, Football, Club}
    >
    > 同样对于这个sentence使用2-gram，可以创建如下特征集：
    >
    > ​		Features = {'Manchester United', 'United is', 'is the', 'the most', 'most successful', 'successful English', 							'English Football', 'Football club'}
    >
    > **In N-grams, the N refers to number of co-occurring words.** N-grams allow us to take the occurrence of the words into account while processing the content of the document. If we look at this sentence we can see that "Manchester United" together provides more information about what is being said in the sentence rather than if you inspect the words "Manchester" and "United" separately.
    >
    > Now, if you look at these N-grams you can see that at least three N-grams convey significant bit of information about the sentence e.g. "Manchester United", "English Football", "Football Club". From these N-grams we can understand that the sentence is about Manchester United that is football club in English football.
    >
    > A set of N-grams can be helpful for thinks like **autocompletion/autocorrect and language models**. Creating an N-gram from a huge corpus of text provides lots of information about **which words typically occur together**, and therfore allows you to **predict what word will come next in a sentence.**
    >
    > N-grams found its primary application in an area of probabilistic language models. As they estimate the probability of the next item in a word sequence.
    >
    > 可用于查重

#### 2、TF-IDF

在词袋模型的基础上进行改进，不仅考虑的word的词频，还考虑了word在所有文档中的出现情况，从而对仅以词频选取特征的方法进行了修正。

- 缺点：
  - 仍然需要创建稀疏矩阵

#### 3、Word2Vec

- 优点：

  - 保留了单词的语义信息

  - 固定维度向量（维度较低）

- 代码：

  ```python
  from gensim.models import Word2Vec
  
  word2vec = Word2Vec(all_words, min_count=2)
  vocabulary = word2vec.wv.vocab
  print(vocabulary)
  
  # 将单词转化为对应的向量
  v1 = word2vec.wv['artificial']
  
  # 查找最相似的TOP 10 的词
  sim_words = word2vec.wv.most_similar('intelligence')
  ```

  

- 总结：

  - 稀疏句向量(独热编码): 

    - BOW() 

    - N-Grams(考虑上下文)

    - TF-IDF(对词频条件进行改进)

      稀疏句向量的维度一般非常高，

  - 稠密词向量

    - Word2Vec

    - Glove

    - FastText



