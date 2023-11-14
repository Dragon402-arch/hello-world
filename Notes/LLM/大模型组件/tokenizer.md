## Tokenization

[讲解1](https://towardsdatascience.com/word-subword-and-character-based-tokenization-know-the-difference-ea0976b64e17)

[讲解2](https://towardsdatascience.com/wordpiece-subword-based-tokenization-algorithm-1fbd14394ed7)

[讲解3](https://blog.floydhub.com/tokenization-nlp/)

**Tokenization** in simple words is the process of splitting a phrase, sentence, paragraph, one or multiple text documents into smaller units. Each of these smaller units is called a **token**. Now, these tokens can be anything — a word, a subword, or even a character.

Tokens are actually the building blocks of NLP and **all the NLP models process raw text at the token level**. **These tokens are used to form the vocabulary, which is a set of unique tokens in a corpus (a dataset in NLP). This vocabulary is then converted into numbers (IDs) and helps us in modeling.**

### 1. Word-based tokenization

#### 1.1 优劣分析

##### 1.1.1 优势

1. word具有含义：由于单词具有含义，可以对学习到的单词嵌入表示进行验证其相似性和相关性。

##### 1.1.2 劣势

1. 词表较大：一种语言包含的words通查都比较大，可以通过词频来限制进行词表的word，但又会带来OOV问题。
2. 存在OOV（Out Of Vocabulary）问题：一般会使用 `[UNK]` token来表示那些不在词表中的word（未登陆词），该做法在NLU任务中影响不大，而在NLG任务中如果生成的文本中出现许多`[UNK] `则会降低生成文本的质量，过滤掉 `[UNK]`后句子可能会变得不通顺。

### 2. Character-based tokenization

Character-based tokenizers split the raw text into individual characters. The logic behind this tokenization is that a language has many different words but has a fixed number of characters. This results in a very small vocabulary.

基于字符形成的分词器将文本拆分为单个字符，其主要思想是一种语言虽然有许多不同的words，但是却只有固定数量的字符characters（如英语中，字母、数字、特殊符号），因此这将形成一个非常小的词表。

基于字符切分的方法首先对文本进行词语（word）切分，然后再对词语进行字符（character）切分，会得到一个字符序列。经过学习，可以得到所有字符的嵌入表示。对构成词语的字符序列表示进行转换（如求和、取平均等操作）后进而可以得到词语对应的嵌入表示。该方法可以很好地解决OOV问题，但是也会带来序列长度的大大增加，而模型处理长序列的难度要比短序列大得多。通常为了达到类似的性能，基于字符的模型需要要比基于词的模型要复杂得多。

#### 2.1 优劣分析

##### 2.1.1 优势

1. 词表较小：与基于word形成的词表相比，基于character形成的词表包含的token数量比较小。
2. 较好地解决OOV问题：即便对于在训练时没有出现过的word，也可以使用已有字符进行表示。

##### 2.1.2 劣势

1. 字符缺乏含义：字符通常不像单词那样携带任何意义或信息。
2. 序列长度大大增加：由于每个word都被切分为character字符序列，因此切分后的序列长度远远大于基于word方式形成的序列长度Each word is split into each character and thus, the tokenized sequence is much longer than the initial raw text.
3. 词嵌入模型复杂度较高

### 3. Subword-based tokenization

[参考](https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0)

子词（subword）切分之前通常需要做词语切分，而词语切分是跟具体语言高度相关的。

Another popular tokenization is subword-based tokenization which is a solution between word and character-based tokenization. The main idea is to solve the issues faced by word-based tokenization (**very large vocabulary size, large number of OOV tokens, and different meaning of very similar words**) and character-based tokenization (**very long sequences and less meaningful individual tokens**).

The subword-based tokenization algorithms uses the following principles.

1. **Do not split the frequently used words into smaller subwords.**
2. **Split the rare words into smaller meaningful subwords.**

**Common words will be tokenized generally as whole words**, e.g. “***the***”, “***at***”, “***and***”, etc., while **rarer words will be broken into smaller chunks** and can be used to create the rest of the words in the relevant dataset.

The subword-based tokenization algorithms generally use a special symbol to indicate which word is the start of the token and which word is the completion of the start of the token. For example, “tokenization” can be split into “token” and “##ization” which indicates that “token” is the start of the word and “##ization” is the completion of the word.

Subword-based tokenization allows the model to have a decent vocabulary size and also be able to learn meaningful context-independent representations. It is even possible for a model to process a word which it has never seen before as the decomposition can lead to known subwords.

优点：

- 子词词表相对较小：与基于word方式的切分相比

  **The larger the vocabulary size the more common words you can tokenize. The smaller the vocabulary size the more subword tokens you need to avoid having to use the <UNK> token.**

  子词词表越大，切分时使用的常见词（也就是完整的word）就越多。子词词表越小，为了避免使用<UNK>token，切分时需要使用的子词就越多。

- 能够学习有意义的上下文无关的表示

- 较好地解决OOV问题：可以处理以前从未见过的word，因为对其进行拆分后可能使其变为多个已有的子词subword，但也会存在出现未登录词的可能性。

关系图：

![Frequency V probability approaches](https://blog.floydhub.com/content/images/2020/02/subword-probabilistic-tokenization.png)

#### 3.1 Byte Pair Encoding (BPE)

##### 3.1.1 使用的模型

- GPT-2
- RoBERTa

##### 3.1.2 BPE算法操作步骤

中文介绍

1. **获取单词以及词频**：在每个单词末尾添加`</w>`符号，获取文本出现的所有单词（word）以及单词对应的词频（frequency）。
2. **获取初始化子词词表**：将所有词语切分成字符序列，获取文本包含的所有字符（包含字母、数字、标点符号、特殊符号、汉字等）以及出现的频数，得到初始化子词词表（subword vocabulary）；（统计语料中所有相邻最小单元组合出现的频率，合并频率最高的组合，形成一个新的子词。）
3. **合并**：将子词词表中共现频数最高（the highest frequency subword pair）的一对子词进行合并（merge），得到一个新的子词。
4. **重新计算子词频数**：将得到的子词添加到子词词表中，然后重新计算子词词表中所有子词的频数，每次合并后子词的频数都会发生变化。
5. **迭代**：重复步骤3、4，直到子词词表中子词数量达到预设值（the predefined vocabulary size）或是设定的最大迭代次数。

[英文介绍](https://blog.floydhub.com/tokenization-nlp/)

There are a few steps to these merging actions：

1. Get the word **count** frequency
2. Get the **initial token count** and frequency (i.e. how many times each character occurs)
3. Merge the **most common byte pair **(find the most common byte pair and merge both into one new token.)
4. Add this to the list of tokens and **recalculate the frequency count** for each token; this will change with each merging step
5. **Repeat** until you have reached your defined token limit or a set number of iterations 

优势：平衡 the vocabulary size 和 token sequence length；高频词最终被表示为一个整体，低频词最终则被切分为多个子词。

##### 3.1.3 BPE算法存在的问题

对一个特定的word进行编码成，可能形成多种不同的子词序列（subword sequence），而这些序列又没有优先级，从而导致相同的输入每次可能会得到不同的编码表示，这会影响学习表示的准确性。可以用如下示例说明：

linear = **li + near** *or* **li + n + ea + r**

algebra = **al + ge + bra** *or* **al + g + e + bra**

在上例中，每个word都存在两种subword序列，对于 linear algebra 则有四种subword序列来编码，也就是相同的输入文本可以用四种方式来编码，同时又无法提供每种编码方式的概率。

For BPE we used the frequency of words to help identify which tokens to merge to create our token set. BPE ensures that the most common words will be represented in the new vocabulary as a single token, while less common words will be broken down into two or more subword tokens. To achieve this, BPE will go through every potential option at each step and pick the tokens to merge based on the highest frequency. In this way it is a greedy algorithm which optimizes for the best solution at each step in its iteration.

**BPE takes a pair of tokens (bytes), looks at the frequency of each pair, and merges the pair which has the highest combined frequency. The process is greedy as it looks for the highest combined frequency at each step.**

#### 3.2 WordPiece

##### 3.2.1 使用的模型

- BERT
- DistilBERT

##### 3.2.2 改进之处

针对BPE算法存在的问题，WordPiece算法对其进行了改进。WordPiece和BPE之间的唯一区别是将subword pair (token pair) 添加到词汇表中的方式，WordPiece在进行合并操作时不再使用词频（count frequency），而是使用似然得分（likelihood）来选择要合并的子词对以及是进行否合并（**WordPiece uses the likelihood rather than count frequency to both choose the pairs to merge and whether to merge them or not.**）。

> **likelihood_score = (freq_of_pair) / (freq_of_first_element × freq_of_second_element)**
>
> By dividing the frequency of the pair by the product of the frequencies of each of its parts, the algorithm prioritizes the merging of pairs where the individual parts are less frequent in the vocabulary. For instance, it won’t necessarily merge `("un", "##able")` even if that pair occurs very frequently in the vocabulary, because the two pairs `"un"` and `"##able"` will likely each appear in a lot of other words and have a high frequency. In contrast, a pair like `("hu", "##gging")` will probably be merged faster (assuming the word “hugging” appears often in the vocabulary) since `"hu"` and `"##gging"` are likely to be less frequent individually.
>
> 不常出现的词被merge了，而常出现的词却不merge，有点奇怪
>
> **WordPiece algorithm trains a language model on the base vocabulary, picks the pair which has the highest likelihood, add this pair to the vocabulary, train the language model on the new vocabulary and repeat the steps repeated until the desired vocabulary size or likelihood threshold is reached.**
>
> WordPiece is also a greedy algorithm that leverages likelihood instead of count frequency to merge the best pair in each iteration but the choice of characters to pair is based on count frequency.
>
> So, it is similar to BPE in terms of choosing characters to pair and similar to Unigram in terms of choosing the best pair to merge.

在构建完成子词词表后，WordPiece方法以左到右最大匹配的方式切分出子词序列（subword sequence），由此实现基于子词对句子的表示。

#### 3.3 Unigram

基于Unigram的子词切分方法假设每个子词（subword）出现的概率相互独立，一个句子可能有多种不同的子词切分结果，其切分成某个子词序列的概率可以表示为其中所有子词出现的概率乘积。寻找句子最优子词切分结果的问题可以转化为句子出现的最大化概率问题，可通过Viterbi算法进行求解。在实际计算时，该算法通过启发式的策略从训练语料中初始化一个种子子词词表，并利用 EM 算法估计每个子词出现的概率。

##### 3.3.1 使用的模型

- XLNet
- ALBERT

##### 3.3.2 改进之处

> Unigram语言模型假定：subword之间是相互独立的，因此subword sequence的概率就是所有subword概率的乘积。
>
> The unigram language model makes an assumption that **each subword occurs independently**, and consequently, the probability of a subword sequence x = ($x_1$, . . . , $x_M$) is formulated as the product of the subword occurrence probabilities $p(x_i)$,
> $$
> P(x) = \prod_{i=1}^Mp(x_i)
> $$
> x表示segmentation candidate，也就是一个候选的子词序列，然后通过MLE极大似然估计找到最大可能的子词序列。
>
> In other words, subword segmentation with the unigram language model can be seen as a probabilsitic mixture of characters, subwords and word segmentations.

操作步骤：

1. 用启发式的方法从训练语料库中初始化一个非常大的种子词词表。
2. 修改词表，使用EM算法优化并计算每个subword token 的概率（也即使用频率估计概率）
3. 计算每个subword token从词表中删除时产生的损失值，也就是似然得分减少的数值，记为该subword token的loss。
4. 根据每个subword token 的损失值对子词进行降序排序，保留前80%-90%的子词，但是总保留单个字符的子词以免出现OOV问题。
5. 重复步骤2-4，直至达到预先定义的词表大小。

操作步骤：

1. Make a reasonably big seed vocabulary from the training corpus.

2. Work out the probability for each subword token with EM algorithm.

3. Work out a loss value which would result if each subwork token were to be dropped.(Compute the loss for each subword , where loss represents how likely the likelihood is reduced when the subword is removed from the current vocabulary)

4. Drop the tokens which have the largest loss value. You can choose a value here, e.g. drop the bottom 10% or 20% of subword tokens based on their loss calculations. **Note you need to keep single characters to be able to deal with out-of-vocabulary words.**（Based on a certain threshold of the loss value, you can then trigger the model to drop the bottom 20-30% of the subword tokens.按照损失阈值删除，还是按照比例20%删除呢）

5. Repeat these steps until you reach your desired final vocabulary size or until there is no change in token numbers after successive iterations.

- Unigram与WordPiece比较

  - 相同之处

    - 使用语言模型：都采用了语言模型来决定子词词表中是否收录某个子词
    - 子词切分方式：切分文本进行表示时都采用从左到右的最大匹配方法

  - 不同之处

    - 子词词表构造方式：

      - WordPiece：先构造字符级别词表，再通过合并操作逐渐往词表中增加新的子词。

      - Unigram：先构造一个大的子词词表（比如收录语料库中所有词），然后再逐渐删除词表中的子词（把罕见的子词切分成两个更常见的子词），直到词表中子词数量达到预定的值。


#### 3.4 SentencePiece

##### 3.4.1 基本思想

子词切分之前通常需要做词语切分，然后再将词语切分为子词序列（subword sequence），而词语切分是跟具体语言高度相关的（英语通常使用空格切分）。**SentencePiece提供了一种与语言类型无关的子词切分方法**，可以在输入的句子上直接做子词切分，无需先做词语切分。在这个方法中，空格不作为词语之间的分隔符，也就是说，空格与其他任何字符同等对待。按照这种方式得到的子词词表，子词中出现空格，或者子词中同时出现字母、数字或者标点符号都是很常见的。

SentencePiece不是和WordPiece同一个层次的概念，只是强调**子词切分之前无需做词语切分**，可以直接从句子开始构造子词词表和进行子词切分，但并没有规定子词词表构造和子词切分的算法。所以即使采用SentencePiece方法，还是需要选择BPE、WordPiece、SentencePiece来做子词词表构造和子词切分的。

- 对比
  - SentencePiece：先对文档进行句子切分，然后直接对句子进行子词切分，跳过了词语（word）切分的过程。
  - BPE、WordPiece、Unigram：先对文档进行句子切分、然后是针对句子进行词语切分，最后针对词语进行子词切分。

##### 3.4.2 代码测试

```python
from transformers import XLNetTokenizer

tokenizer = XLNetTokenizer.from_pretrained("./xlnet-base-cased")

text1 = "我们都有一个家，你知道吗？"

text2 = "Long long ago, do you have a dream?"

print(tokenizer.tokenize(text1))
print(tokenizer.tokenize(text2))


"""
输出：
['▁', '我们都有一个家', ',', '你知道吗', '?']
['▁Long', '▁long', '▁ago', ',', '▁do', '▁you', '▁have', '▁a', '▁dream', '?']
"""
```

1. SentencePiece会将空格编码为‘_’， 如'New York' 会转化为['▁New', '▁York']，这也是为了能够处理多语言问题，比如英文解码时有空格，而中文没有。
2. 以unicode方式编码字符，将所有的输入（英文、中文等不同语言）都转化为unicode字符，解决了多语言编码方式不同的问题。



#### 3.5 总结

1. **BPE:** Just uses the frequency of occurrences to identify the best match at every iteration until it reaches the predefined vocabulary size.
2. **WordPiece**: Similar to BPE and uses frequency occurrences to identify potential merges but makes the final decision based on the likelihood of the merged token
3. **Unigram**: A fully probabilistic model which does not use frequency occurrences. Instead, it trains a LM using a probabilistic model, removing the token which improves the overall likelihood the least and then starting over until it reaches the final token limit.



- [Training BPE, WordPiece, and Unigram Tokenizers from Scratch using Hugging Face | by Harshit Tyagi | Towards Data Science](https://towardsdatascience.com/training-bpe-wordpiece-and-unigram-tokenizers-from-scratch-using-hugging-face-3dd174850713)









