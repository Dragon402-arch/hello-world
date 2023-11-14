- torch.masked_select()

- torch.index_select()

  示例:

  ```python
  keywords_index = torch.tensor([1, 3, 5, 7, 9])
  keywords_embed = torch.index_select(torch.rand(12, 1, 768), 0, keywords_index)
  print(keywords_embed.shape)
  
  # 输出：torch.Size([5, 1, 768])
  ```

  等价操作：

  ```python
  from operator import itemgetter
  
  keywords_index = torch.tensor([1, 3, 5, 7, 9])
  keywords_embed= itemgetter(keywords_index)(torch.rand(12, 1, 768))
  print(keywords_embed.shape)
  
  # 输出：torch.Size([5, 1, 768])
  ```

  

- view 和 reshape的区别

  如果对 tensor 调用过`transpose`,`permute`等操作的话会使该 tensor 在内存中变得不再连续，`.view()`方法只能改变连续的(**contiguous**)张量，因此通常需要先调用`.contiguous()`方法。

  ```python
  x  = torch.rand(2,12,32,64)
  x = x.permute(0, 2, 1, 3).contiguous() # (2,32,12,64)
  new_x_shape = (2,32,768)
  x = x.view(*new_x_shape)
  print(x.shape) # (2,32,768)
  ```

- `torch.enisum()`

  - 矩阵乘法

    方式一

    ```python
    >>> import torch
    >>> A = torch.randn(3, 4)
    >>> B = torch.randn(4, 5)
    >>> C = torch.einsum('ik,kj->ij', A, B)
    >>> C.shape
    torch.Size([3, 5])
    ```

    ```python
    >>> As = torch.randn(3,2,5)
    >>> Bs = torch.randn(3,5,4)
    >>> torch.einsum('bij,bjk->bik', As, Bs)
    ```

    方式二

    ```python
    >>> A = torch.randn(3, 4)
    >>> B = torch.randn(5, 4)
    >>> C = torch.einsum('ik,jk->ij', A, B)
    >>> C.shape
    torch.Size([3, 5])
    ```

  - 所有元素求和

    ```python
    >>> C = torch.einsum('ij->j', A)
    >>> C
    tensor([-0.4369,  1.1548,  0.8594, -0.0198])
    >>> C.shape
    torch.Size([4])
    ```

  - 行或列求和

    ```python
    C = torch.einsum('ij->j', A)
    C = torch.einsum('ij->i', A)
    ```

  - 转置

    ```python
     C = torch.einsum('ij->ji', A)
    ```

  - permute

    ```python
     # batch permute
    >>> A = torch.randn(2, 3, 4, 5)
    >>> torch.einsum('...ij->...ji', A).shape
    torch.Size([2, 3, 5, 4])
    ```

  - 外积

    ```python
    # outer product
    >>> x = torch.randn(5)
    >>> y = torch.randn(4)
    >>> torch.einsum('i,j->ij', x, y)
    ```

    

- 

