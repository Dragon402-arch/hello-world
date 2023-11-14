## BERT

> 观点：I personally feel that to fully understand “what it actually is”, the best way is to **code it from scratch** to avoid leaving any single detail behind

### 1. WordPiece Tokenization

The initial stage of creating a fresh BERT model involves **training a new tokenizer**. Tokenization is the process of breaking down a text into smaller units called “**tokens**”, which are then converted into a numerical representation.BERT employs a **WordPiece** tokenizer, which can split a single word into multiple tokens. For instance：

```markdown
 “I like surfboarding!” → 
 [‘[CLS]’, ‘i’, ‘like’, ‘surf’, ‘##board’, ‘##ing’, ‘!’, ‘[SEP]’] → 
 [1, 48, 250, 4033, 3588, 154, 5, 2]
```

WordPiece computes a score for each pair, using the following:

> **score = (freq_of_pair) / (freq_of_first_element × freq_of_second_element)**
>
> By dividing the frequency of the pair by the product of the frequencies of each of its parts, the algorithm prioritizes the merging of pairs where the individual parts are less frequent in the vocabulary. For instance, it won’t necessarily merge `("un", "##able")` even if that pair occurs very frequently in the vocabulary, because the two pairs `"un"` and `"##able"` will likely each appear in a lot of other words and have a high frequency. In contrast, a pair like `("hu", "##gging")` will probably be merged faster (assuming the word “hugging” appears often in the vocabulary) since `"hu"` and `"##gging"` are likely to be less frequent individually.

#### 1.1 special tokens

To specifically highlight these special tokens for BERT:

- `CLS` stands for classification. It serves as the the Start of Sentence (SOS) and represent the meaning of the entire sentence.
- `SEP` serves as End of Sentence (EOS) and also the separation token between first and second sentences.
- `PAD`to be added into sentences so that all of them would be in equal length. During the training process, please note that the [PAD] token with id of 0 will not contribute to the gradient .
- `MASK` for word replacement during masked language prediction
- `UNK` serves as a replacement for token if it’s not being found in the tokenizer’s vocab.

### 2 Input Embeddings

#### 2.1  Embedding Type

1. **Token embeddings**: A [CLS] token is added to the input word tokens at the beginning of the first sentence and a [SEP] token is inserted at the end of each sentence.

2. **Segment embeddings**: A marker indicating Sentence A or Sentence B is added to each token. This allows the encoder to distinguish between sentences.

3. **Positional embeddings**: A positional embedding is added to each token to indicate its position in the sentence.

   ![input_embed](D:\Typora\Notes\NLP\经典模型\input_embed.png)

### 2. Pre-Training Strategy 

#### 2.1 Masked Language Model (MLM)

The simple idea by masking 15% of the words with `MASK ` token and predict them. Yet, there is a problem with this masking approach as the model only tries to predict when the [MASK] token is present in the input, while we want the model to try to predict the correct tokens regardless of what token is present in the input. To deal with this issue, out of the 15% of the tokens selected for masking:

- 80% of the tokens are actually replaced with the token [MASK].
- 10% of the time tokens are replaced with a random token.
- 10% of the time tokens are left unchanged.

**代码实现1**：[出处 | 官方源码实现](https://github.com/jiesutd/pytorch-pretrained-BERT/blob/master/examples/lm_finetuning/pregenerate_training_data.py)

```python
import random

from transformers import BertTokenizer


def create_masked_lm_predictions(
        tokens, masked_lm_prob=0.15, max_predictions_per_seq=100, vocab_list=None
):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables.
    Args:
        tokens:[CLS] + seq_1_tokens + [SEP] + seq_2_tokens + [SEP] 或者 [CLS] + seq_tokens + [SEP]
        masked_lm_prob(float):0.15
        max_predictions_per_seq(int):100; 一个句子对中最大可以被MASK的token个数,
        vocab_list(list): 词汇列表
    Returns
        tokens:原token序列被MASK之后得到的token序列,
        mask_indices:被MASK掉token在序列中的索引值,
        masked_token_labels：被MASK掉token的原来值，充当模型学习的目标
    """
    candidate_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        candidate_indices.append(i)

    num_to_mask = min(
        max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob)))
    )
    random.shuffle(candidate_indices)
    mask_indices = sorted(random.sample(candidate_indices, num_to_mask))
    print(num_to_mask)
    print(mask_indices)
    # 保存被mask前token的真实值
    masked_token_labels = []
    for index in mask_indices:
        # 80% of the time, replace with [MASK]
        if random.random() < 0.8:  # 8:2(1:1)
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = random.choice(vocab_list)
        masked_token_labels.append(tokens[index])
        # Once we've saved the true label for that token, we can overwrite it with the masked version
        tokens[index] = masked_token

    return tokens, mask_indices, masked_token_labels


def test():
    pretrained_model_path = r"E:\pretrained_model\chinese_wwm_ext_pytorch"
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    text_a = "我们都有一个家，名字叫中国，兄弟姐妹都很多，景色也不错."
    text_b = "家里盘着两条龙，是长江与黄河呀，还有珠穆朗玛峰儿，是最高山坡。"
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = tokenizer.tokenize(text_b)
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
    print(tokens)

    vocab_list = list(tokenizer.get_vocab().keys())

    tokens, mask_indices, masked_token_labels = create_masked_lm_predictions(
        tokens, masked_lm_prob=0.15, max_predictions_per_seq=100, vocab_list=vocab_list
    )
    print(tokens)
    print(masked_token_labels)

test()


"""
['[CLS]', '我', '们', '都', '有', '一', '个', '家', '，', '名', '字', '叫', '中', '国', '，', '兄', '弟', '姐', '妹', '都', '很', '多', '，', '景', '色', '也', '不', '错', '.', '[SEP]', '家', '里', '盘', '着', '两', '条', '龙', '，', '是', '长', '江', '与', '黄', '河', '呀', '，', '还', '有', '珠', '穆', '朗', '玛', '峰', '儿', '，', '是', '最', '高', '山', '坡', '。', '[SEP]']
9
[3, 5, 14, 17, 20, 36, 39, 49, 57]
['[CLS]', '我', '们', '[MASK]', '有', '[MASK]', '个', '家', '，', '名', '字', '叫', '中', '国', '[MASK]', '兄', '弟', '[MASK]', '妹', '都', '[MASK]', '多', '，', '景', '色', '也', '不', '错', '.', '[SEP]', '家', '里', '盘', '着', '两', '条', '[MASK]', '，', '是', '[MASK]', '江', '与', '黄', '河', '呀', '，', '还', '有', '珠', '[MASK]', '朗', '玛', '峰', '儿', '，', '是', '最', '[MASK]', '山', '坡', '。', '[SEP]']
['都', '一', '，', '姐', '很', '龙', '长', '穆', '高']

"""
```

当句子超过之指定的最大长度时，对句子对进行截断的原则，左右两侧都会截断。

```python
def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    # 摘自Google's BERT 的代码仓库
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()
```

代码实现2

```python
class BERTDataset(Dataset):
    def __init__(self, data_pair, tokenizer, seq_len=64):

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_lines = len(data_pair)
        self.lines = data_pair

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):

        # Step 1: get random sentence pair, either negative or positive (saved as is_next_label)
        t1, t2, is_next_label = self.get_sent(item)

        # Step 2: replace random words in sentence with mask / random words
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # Step 3: Adding CLS and SEP tokens to the start and end of sentences
        # Adding PAD token for labels
        input_ids_1 = [self.tokenizer.vocab['[CLS]']] + t1_random + [self.tokenizer.vocab['[SEP]']]
        input_ids_2 = t2_random + [self.tokenizer.vocab['[SEP]']]

        # TODO 如果ignore_label的值为[PAD],则是正常的，否则应该对label填充-100，
        seq_1_label = [self.tokenizer.vocab['[PAD]']] + t1_label + [self.tokenizer.vocab['[PAD]']]
        seq_2_label = t2_label + [self.tokenizer.vocab['[PAD]']]

        # Step 4: combine sentence 1 and 2 as one input adding PAD tokens to make the sentence same length as seq_len
        segment_ids = ([1 for _ in range(len(input_ids_1))] + [2 for _ in range(len(input_ids_2))])[:self.seq_len]
        input_ids = (input_ids_1 + input_ids_1)[:self.seq_len]
        labels = (seq_1_label + seq_2_label)[:self.seq_len]

        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(input_ids))]

        input_ids.extend(padding)
        segment_ids.extend(padding)
        labels.extend(padding)

        output = {"bert_input": input_ids,
                  "bert_label": labels,
                  "segment_label": segment_ids,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        output = []

        # 15% of the tokens would be replaced
        for i, token in enumerate(tokens):
            prob = random.random()

            # remove cls and sep token
            token_id = self.tokenizer(token)['input_ids'][1:-1]

            if prob < 0.15:
                prob /= 0.15

                # 80% chance change token to mask token
                if prob < 0.8:
                    for i in range(len(token_id)):
                        output.append(self.tokenizer.vocab['[MASK]'])

                # 10% chance change token to random token
                elif prob < 0.9:
                    for i in range(len(token_id)):
                        output.append(random.randrange(len(self.tokenizer.vocab)))

                # 10% chance change token to current token
                else:
                    output.append(token_id)

                output_label.append(token_id)

            else:
                output.append(token_id)
                for i in range(len(token_id)):
                    output_label.append(0)

        # flattening
        output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
        output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
        assert len(output) == len(output_label)
        return output, output_label

    def get_sent(self, index):
        """return random sentence pair"""
        t1, t2 = self.get_corpus_line(index)

        # negative or positive pair, for next sentence prediction
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        """return sentence pair"""
        return self.lines[item][0], self.lines[item][1]

    def get_random_line(self):
        """return random single sentence"""
        return self.lines[random.randrange(len(self.lines))][1]
```

itertools.chain

```python
output = [106, 109, [108, 256], 365, 562]

out = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
print(out)

"""
输出:
[106, 109, 108, 256, 365, 562]
"""
```

It randomly masks words in the sentence and then it tries to predict them. Masking means that the model looks in both directions and it uses the full context of the sentence, both left and right surroundings, in order to predict the masked word. Unlike the previous language models, it takes both the previous and next tokens into account at the **same time.** The existing combined left-to-right and right-to-left LSTM based models were missing this “same-time part”. （BERT：并行、双向、同时；BiLSTM：串行、双向、不同时）

#### 2.2 Next Sentence Prediction (NSP)

The NSP task forces the model to understand the relationship between two sentences. In this task, BERT is required to predict whether the second sentence is related to the first one. During training, the model is fed with 50% of connected sentences and another half with random sentence sequence.

During training the model is fed with two input sentences at a time such that:

- 50% of the time the second sentence comes after the first one.
- 50% of the time it is a a random sentence from the full corpus.

### 学习链接

- [Mastering BERT Model: Building it from Scratch with Pytorch](https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891)：包含代码

- [BERT Explained: A Complete Guide with Theory and Tutorial](https://medium.com/@samia.khalid/bert-explained-a-complete-guide-with-theory-and-tutorial-3ac9ebc8fa7c) ：不含代码

