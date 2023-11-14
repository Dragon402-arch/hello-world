- [CPython and Cython](https://medium.com/@elhayefrat/python-cpython-e88e975e80cd)

- Python 代码执行内部过程

  Python implementation in C (CPython) is not 100% complied, and also not 100% interpreted. **There is both compilation and interpretation in the process of running a Python script.** To make this clear, let’s see the steps of running a Python script:

  1. Compiling source code using CPython to generate **bytecode**
  2. Interpreting the bytecode in a CPython interpreter
  3. Running the output of the CPython interpreter in a CPython virtual machine

  Compilation takes place when CPython compiles the source code (.py file) to generate the CPython bytecode (.pyc file). The CPython bytecode (.pyc file) is then interpreted using a CPython interpreter, and the output runs in a CPython virtual machine. According to the above steps, the process of running a Python script involves both compilation and interpretation.

  **The CPython compiler generates the bytecode just once, but the interpreter is called each time the code runs.**

  通常代码在执行后会出现一个名为 `__pycache__ ` 的文件夹，在该文件夹下存放着诸如 `entity_normalization.cpython-36.pyc` 的文件，这些文件是Python代码经过编译之后生成的字节码文件，然后使用CPython 解释器解释字节码文件从而产生最终的输出。由于是”一次编译，多次解释“的机制，因此同样的代码在第一次执行时会花费较长的时间，而在后续执行同样的代码时相比于第一次的时间会有所减少，这是因为不需要再进行编译步骤的缘故。

  ![image-20220418111735787](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20220418111735787.png)

  **In summary, using a compiler speeds up the process but an interpreter makes the code cross-platform. So, a reason why Python is slower than C is that an interpreter is used. Remember that the compiler just runs once but the interpreter runs each time the code is executed.**

- 

