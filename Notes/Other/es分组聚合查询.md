```json
{
    "query": {"bool": {"must": [], "must_not": [], "should": []}},
    "from": 1,
    "size": 10,
    "sort": [],
    "aggs": {"groupby": {"terms": {"field": "OUT_VERTEX_ID","size":10, "min_doc_count":0},
                         "aggs":{"my_top_hits":{"top_hits":{"_source": {"includes": ["CC","GX_DATE"]},"size":3}},
                                 "frequency": {"cardinality" : {"script":"doc['CC'].value + '#' + doc['GX_DATE'].value"}},
                                  "queryCs":{"bucket_selector": {"buckets_path": {"cs":"frequency"},"script": "params.cs > 10"}}
                                    }
                                }
                            }
                    }
```



### 1、分组聚合时存在两种情况：

- 按照一个字段进行分组聚合，可以写为：


```json
{"terms": {"field": "OUT_VERTEX_ID","size":10, "min_doc_count":0}}
```

- 按照多个字段进行分组聚合，可以写为：


```json
{"terms":{ "script": "doc['age'].value + '#' +doc['city'].value","size":10, "min_doc_count":0 }}
```

### 2、分组之后，组内进行去重计数，存在两种情况：

组内不去重直接计数，使用 value_count 代替 cardinality 即可。value_count:计数 ; cardinality: 去重计数

- 按照一个字段进行组内去重计数：

```json
{"cardinality": {"field": "GX_DATE"}}
```

- 按照多个字段进行组内去重计数：


```json
{"cardinality" : {"script":"doc['CC'].value + '#' + doc['GX_DATE'].value"}}
```

### 3、分组之后，进行过滤

```json
{
    "bucket_selector": {
        "buckets_path": {
            "my_var1": "the_sum", 
            "my_var2": "the_value_count"
        },
        "script": "params.my_var1 > params.my_var2"
    }
}
```

[es官方教程](https://www.elastic.co/guide/en/elasticsearch/reference/5.6/search-aggregations-metrics-cardinality-aggregation.html)

[知乎详细讲解](https://zhuanlan.zhihu.com/p/183816335)

text 类型的字符串可以使用match,match_phrase, 分词存储，匹配时会进行分词。

keyword 类型的字符串只能使用wildcard进行匹配，而无法使用match,match_phrase进行匹配查询，因为其不进行分词，整体存储。

分组和去重时只能使用keyword类型的字段。

- 分词器测试：

  ```
  http://192.168.52.34:9296/intent_search/_analyze
  ```

  输入文本：

  ```json
  {
    "text":"中华人民共和国国徽",
    "analyzer":"ik_smart"
  }
  
  {
    "text":"中华人民共和国国徽",
    "analyzer":"ik_max_word"
  }
  ```

  

