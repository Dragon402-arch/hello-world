#### 细节问题列表

1. batch_size 的变化情况

   由于使用多块GPU进行并行训练，相当于加大了batch_size，从而起到了加速训练的效果。因此在生成data_loader之前需要进行如下处理：

   ```python
   args.batch_size = int(args.batch_size / ngpus_per_node)
   ```

   经过该处理后的 batch_size 参数是每个GPU实际使用的大小，而直接传入的参数则是模型训练实际用到的batch_size。

2. learning_rate的变化情况

   A key aspect of using large batch sizes involves scaling the learning rate. A general rule of thumb is to follow a Linear Scaling Rule. This means that **when the batch size increases by a factor of K the learning rate must also increase by a factor of K.**这意味着当batch_size 增加K倍时，学习率也必须增加K倍。

   [代码出处](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/train_multi_GPU/train_multi_gpu_using_spawn.py)

   [理论出处](https://towardsdatascience.com/why-parallelized-training-might-not-be-working-for-you-4c01f606ef2c)

   ```python
   args.learning_rate *= ngpus_per_node 
   ```

3. drop_last带来的截断和填充，默认为False

   Suppose K=3, and the length of dataset is 10. We must understand that **DistributedSampler imposes even partition of indices.**

   - If we set `drop_last=False` when defining DistributedSampler, it will automatically **pad.** For example, it splits indices [0,1,2,3,4,5,6,7,8,9] to [0,3,6,9] when *rank*=1, [0,4,7,0] when *rank*=2, and [2,5,8,0] when *rank*=3. As you can see, such padding may cause issues because the padded 0 **is** a data record.
   - Otherwise, it will **strip off the trailing elements**. For example, it splits the indices to [0,3,6] at *rank*=1, [1,4,7] at *rank*=2, and [2,5,8] at *rank*=3. In this case, it tailored 9 to make the indice number divisible by world_size.

4. 两个特殊方法的使用

   - gather

     ```python
     #  -*- coding: utf-8 -*-
     import os
     from argparse import ArgumentParser
     
     import torch
     import torch.distributed as dist
     import torch.multiprocessing as mp
     
     
     def init_dist(rank, world_size):
         os.environ['MASTER_ADDR'] = 'localhost'
         os.environ['MASTER_PORT'] = '12355'
         dist.init_process_group(
             "nccl", rank=rank, init_method="tcp://127.0.0.1:3457", world_size=world_size
         )
     
     
     def get_args():
         parser = ArgumentParser()
         # GPU配置参数
         parser.add_argument(
             "--gpu_devices", type=int, nargs="+", default=[1, 2], help="使用GPU的ID列表"
         )  # 该变量传递的是一个列表参数
         parser.add_argument(
             "--target_device", type=int, default=1, help="指定数据到的目标设备"
         )
         args = parser.parse_args()
         return args
     
     
     def main_worker(nprocs, args, ngpus_per_node):
         """
         Args:
             nprocs: nprocs 的取值是所使用的GPU索引数，可能的取值为0,1,2,3
         """
         init_dist(rank=nprocs, world_size=ngpus_per_node)
         assert args.target_device in args.gpu_devices
     
         # 将进程与指定的GPU进行对应，否则就会选取(0,1)/(0,1,2)的GPU组合，对应后可以选到(1,2)这种组合。
         gpu_idx = args.gpu_devices[nprocs]
         torch.cuda.set_device(gpu_idx)
         data = torch.arange(2, dtype=torch.long).to(gpu_idx) + 1 + 2 * gpu_idx
         print(data)
         output_list = [torch.zeros(2, dtype=torch.long).to(gpu_idx) for _ in range(dist.get_world_size())]
         dist.gather(data, output_list if gpu_idx == args.target_device else None,
                     dst=args.gpu_devices.index(args.target_device))
         print(f"rank: {gpu_idx}, data: {output_list}")
     
     
     def do_test(args):
         ngpus_per_node = len(args.gpu_devices)  # torch.cuda.device_count()
         mp.spawn(
             main_worker,
             args=(args, ngpus_per_node),
             nprocs=ngpus_per_node,
         )  # nprocs:number process
     
     
     if __name__ == "__main__":
         args = get_args()
         do_test(args)
         
     
         
         
     """
     输出结果：
     tensor([5, 6], device='cuda:2')
     tensor([3, 4], device='cuda:1')
     rank: 1, data: [tensor([3, 4], device='cuda:1'), tensor([5, 6], device='cuda:1')]
     rank: 2, data: [tensor([0, 0], device='cuda:2'), tensor([0, 0], device='cuda:2')]
     """    
     
     ```

     `gather(tensor, gather_list=None, dst=0)` 该方法的 `gather_list` 参数在选定的目标设备上不能为 None，因此代码的45行有添加条件进行判断。关于 `dst ` 参数，必须和前面指定的目标设备保持一致，而且dst的取值从0开始的，比如使用的GPU ID为1、2，此时dst对应的取值就是0、1，必须保持对应关系。

     `init_process_group(backend="nccl",init_method="tcp://127.0.0.1:3457",world_size=world_size,rank=rank)`

     - world_size (int, optional): Number of processes participating in the job.

       参与任务的进程数量，比如使用3块GPU，则进程数据就是3。

     - rank (int, optional): Rank of the current process (it should be a number between 0 and ``world_size``-1).

       当前进程的排名，也就是第一个、第二个、第三个进程。

   - all_gather 

     ```python
     import torch
     import torch.distributed as dist
     
     def gather_all_data(true_labels, pred_labels, device):
         # 先将数据转换为张量形式，并分别放到每个GPU上
         true_labels = torch.as_tensor(true_labels, dtype=torch.long, device=device)
         pred_labels = torch.as_tensor(pred_labels, dtype=torch.long, device=device)
         # 创建输出格式列表，作为一个dist.all_gather()的一个参数。
         all_true_labels = [
             torch.ones_like(true_labels) for _ in range(dist.get_world_size())
         ]
         all_pred_labels = [
             torch.ones_like(pred_labels) for _ in range(dist.get_world_size())
         ]
         # 经过函数处理之后，每个GPU上都存储了所有的数据，我们可以选取其中一份计算模型性能指标。
         dist.all_gather(tensor_list=all_true_labels,tensor=true_labels)
         dist.all_gather(tensor_list=all_pred_labels,tensor=pred_labels)
         # dist.gather()
         # 将张量列表拼接为一个张量，便于进行指标计算。
         true_labels = torch.cat(all_true_labels, dim=0)
         pred_labels = torch.cat(all_pred_labels, dim=0)
     
         return true_labels, pred_labels
     ```

     

5. GPT微调参考代码

   https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/simple_thu_chatglm6b

   https://github.com/LC1332/Chinese-alpaca-lora

   https://github.com/mymusise/ChatGLM-Tuning

   

   https://github.com/THUDM/ChatGLM-6B

   https://github.com/HarderThenHarder/transformers_tasks/tree/main/LLM

   https://www.kaggle.com/code/tuckerarrants/text-generation-with-huggingface-gpt2/notebook

   http://mohitmayank.com/a_lazy_data_science_guide/natural_language_processing/GPTs/
   
   https://blog.csdn.net/phycoding/article/details/129884586


https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o?usp=sharing#scrollTo=otj46qRbtpnd

1. fp16 + loss scale

   https://www.jianshu.com/p/58a68fa69332

2. 

   https://www.pinecone.io/learn/batch-layer-normalization/

   https://towardsdatascience.com/different-normalization-layers-in-deep-learning-1a7214ff71d6

   

- GPT2

  https://github.com/Morizeyao/GPT2-Chinese

  http://mohitmayank.com/a_lazy_data_science_guide/natural_language_processing/GPTs/#finetuning-gpt-2-for-sentiment-classification

  https://gmihaila.github.io/tutorial_notebooks/pretrain_transformers_pytorch/

  

- <eop>代表生成模型中的结束标记，表示生成的文本已经结束。在文本生成任务中，生成模型需要根据输入的前缀生成一段完整的文本，<eop>标记则表示生成的文本已经到达了预设的长度或者遇到了特定的结束标记。

- 

- `self.half()` is equivalent to `self.to(torch.float16)`. See [`to()`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to).

  - Parameters:

- 

- 链接

  https://medium.com/grabngoinfo/transfer-learning-for-text-classification-using-hugging-face-transformers-trainer-13407187cf89

  https://www.mlexpert.io/machine-learning/tutorials/alpaca-and-llama-inference

  https://101.dev/t/24-gb-rlhf-20b-llms/922

  https://zhuanlan.zhihu.com/p/604338403

  https://github.com/huggingface/transformers/issues/21736

  https://github.com/huggingface/peft/issues/131

  https://github.com/HarderThenHarder/transformers_tasks/tree/main/LLM/finetune

  

- 

- 

- nn.DataParallel（不推荐使用）

  - 支持单机多卡，即支持使用一台服务器上的多个GPU。

  - 很容易使用，只需要加一行代码，但是速度慢（主要原因是它采用parameter server 模式，一张主卡作为reducer，负载不均衡，主卡成为训练瓶颈），在主GPU上进行梯度计算和更新，再将参数给其他gpu。

    ```python
    model = nn.DataParallel(model)
    ```

    

- nn.DistributedDataParallel （DDP）

  - 支持单机多卡和多机多卡

  - 实现细节：

    复制模型到多个GPU上，每个GPU通过一个进程来控制，进程之间互相通信，只有梯度信息是需要不同进程gpu之间通信，各个进程进行梯度汇总平均，然后传给其他进程，各个进程更新参数。所以瓶颈限制没有那么严重，为pytorch推荐的多GPU方式。

    

    在训练时，每个进程/GPU load 自己的minibatch数据（所以要用distributedsampler), 每个GPU做自己独立的前向运算，反向传播时梯度all-reduce在各个GPU之间，各个节点得到平均梯度，保证各个GPU上的模型权重同步。

    多进程之间同步信息通信是通过 distributed.init_process_group实现。

  - DDP相关概念

    1. group: 进程组 （通过 init_process_group() 对进程组初始化）
    2. world_size: 总的进程数 （通过 get_world_size() 得到）
    3. rank：当前进程号，主机master节点rank=0
    4. local_*rank: 当前进程对应的GPU号 （通过get_rank() 可以得到每台机器上的local_rank*）

    （ps：2台8卡机器进行DDP训练，init_process_*group() 初始化之后，get_world_size() 得到的是16， get_rank() 得到0-8*）

   python -W ignore file.py

  

  加大batch_size可以成倍降低训练时间，

   

  python -W ignore multi_gpu_train.py --gpu_devices 0 1 2 3 --batch_size 192 --lr 5e-5 --num_workers 8

  

  

  python -W ignore run.py --gpu_devices 0 1 2 3 --batch_size 4

  

  

  **# 在一个GPU上进行训练的时候，使用交叉熵损失函数计算损失的时候其默认的就是计算batch上平均损失。**

  **# By default,the losses are averaged over each loss element in the batch.**

  而在使用多个GPU的情况下，如使用4个，batch_size = 128,此时每个gpu上在每个batch进行训练是时分配到的数据是32个，

  一个gpu开启一个进程process，然后每个GPU分别计算各自数据的平均损失，也就会出现4个不同的损失，然后每个loss再分别使用backward()进行反向传播，

  为了和单个GPU计算loss方式保持一致，可以取4个loss的平均值，然后再将平均值分配到4个gpu上，再使用相同的值对多个GPU上的每个数据进行方向传播，

  此时，就和单个GPU训练方式保持了一致。

  torch.distributed.all_reduce(input_tensor, reduce_op=ReduceOp.SUM) 

  Collect the input_tensor from all devices and reduce them using the specified reduce operation such as sum, mean, etc.

   The final  result is copied to all devices.

  

   对于这样一种场景:如果在一个epoch内训练的过程中就直接直接打印输出每个batch上准确率，那么此时就需要把每个GPU上预测正确的个数加起来，

   然后再除以batch_size,即可得到每个batch_size上的准确率，而且各个GPU上的准确率是一致的，否则就会输出得到三个。

  

   为啥我的训练会得到三个损失和三个准确率呢？

   原因如下：

   关于准确率：以一个batch为例，4个gpu,batch_size = 128,然后每个gpu上的数据为32，然后预测结果分别是16,17,15,18，

   正确的做法是16+15+17+18/32*4,而我们的做法是16/32,17/32,15/32,18/32，因此会得到四个值。

