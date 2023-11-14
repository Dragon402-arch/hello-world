## Python进阶

1. 对象的比较与复制
2. 参数的传递
3. 迭代器
4. 生成器
5. 装饰器
6. 元类
7. 操作符重载
8. 上下文管理器
9. 并发编程
10. 全局解释器锁
11. 垃圾回收机制
12. Python与其他语言（C++）的混合使用

#### 后端开发（服务器端开发）

- Django 框架
- Flask 轻量级框架
- 用户登录认证
- 缓存
- 端到端监控
- 单元测试

#### 进阶问题

- Python 中的协程和线程有什么区别？
- 生成器如何进化成协程？
- 并发编程中的 future 和 asyncio 有什么关系？
- 如何写出线程安全的高性能代码呢？

A decorator in Python is a function that takes another function as its argument, and returns yet another function. Decorators can be extremely useful as they allow the extension of an existing function, without any modification to the original function source code.

协同程序、协程coroutine

- 生成器（Generator）
- 协程（Coroutine）：一种特殊类型的函数。受益于在整个生命周期保存数据的能力，与函数不同，协程有几个可以暂停和恢复执行的入口点。
- 线程（Thread）

操作系统基本特征：

- 并发：在一段时间内，宏观上有多个程序同时执行，微观上仍是分时交替执行。

  - 并行：同一时刻多个程序同时执行，而并发则是在同一时刻只有一个程序在执行。

- 共享

  - 互斥共享：一段时间内只允许一个进程对资源进行访问。

  - 同时访问：一段时间内允许多个进程“同时”对资源进行访问，仍是分时交替访问。

    主要区别在于，前者只能在一个进程访问结束，才能开始下一个，而后者则是一个进程访问一部分，允许下一个进程也访问一部分，但是在某一个时刻都是只有一个进程在访问。 

- 异步：允许多个程序并发执行，但是由于资源有限，一个进程的执行不是一次执行完毕的，而是多次执行才能实现的，这就是进程的异步性。

- 虚拟

