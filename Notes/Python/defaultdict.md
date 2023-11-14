## defaultdict

##### 1、默认值是 list

Python中通过Key访问字典中对应的Value值，当Key不存在时，会触发‘KeyError’异常。为了避免这种情况的发生，可以使用collections类中的defaultdict()方法来为字典提供默认值。

```python
from collections import defaultdict

test_dict = defaultdict(list)
data = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
for key, value in data:
    test_dict[key].append(value)
print(test_dict)

# 输出：defaultdict(<class 'list'>, {'yellow': [1, 3], 'blue': [2, 4], 'red': [1]})

print(test_dict["gel"])
# 输出：[]


# 字典自带的setdefault情况

new_dict = {}

for key, value in data:
    new_dict.setdefault(key, []).append(value)
print(new_dict)
# 输出：{'yellow': [1, 3], 'blue': [2, 4], 'red': [1]}

print(new_dict('gel'))
'''
输出：
Traceback (most recent call last):
  File "/home/chinaoly/lis/intentSearch/main.py", line 22, in <module>
    print(new_dict["gel"])
KeyError: 'gel'
'''
```

##### 2、默认值是 int

```python
from collections import defaultdict

# 用于计数
count_dict = defaultdict(int)  # 默认的整数值为0
text = 'mississippi'
for i in text:
    count_dict[i] += 1
print(count_dict)

# 输出：defaultdict(<class 'int'>, {'m': 1, 'i': 4, 's': 4, 'p': 2})

```

##### 3、默认值是 set

```python
from collections import defaultdict

test_dict = defaultdict(set)
data = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
for key, value in data:
    test_dict[key].add(value)
print(test_dict)

# 输出： defaultdict(<class 'set'>, {'yellow': {1, 3}, 'blue': {2, 4}, 'red': {1}})
```