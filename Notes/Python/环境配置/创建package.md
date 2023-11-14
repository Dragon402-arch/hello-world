- [讲解](https://towardsdatascience.com/how-to-package-your-python-code-df5a7739ab2e)

- 常见安装包文件格式：

  示例：Downloading Pillow-9.1.0-cp37-cp37m-win_amd64.whl (3.3 MB)

  使用 pip 安装方式 都是安装的whl格式的文件

- package file structure：

  ````
  packaging_tutorial/
  ├── LICENSE
  ├── pyproject.toml
  ├── README.md
  ├── setup.cfg
  ├── src/
  │   └── example_package/
  │       ├── __init__.py
  │       └── example.py
  └── tests/ 测试脚本存放文件夹
  ````

  - There are two types of metadata: static and dynamic.
    - Static metadata (`setup.cfg`): guaranteed to be the same every time.  推荐使用
    - Dynamic metadata (`setup.py`): possibly non-deterministic. 老版本使用

  

- 安装package（Local Install）

  ```shell
  # 进入package所在目录
  cd \Pycharm\textProcess\aesthetic_ascii
  
  # 安装package
  (torch_venv) E:\Pycharm\textProcess\aesthetic_ascii>pip install .
  
  Processing e:\pycharm\textprocess\aesthetic_ascii
    Installing build dependencies ... done
    Getting requirements to build wheel ... done
    Preparing metadata (pyproject.toml) ... done
  Building wheels for collected packages: aesthetic-ascii
    Building wheel for aesthetic-ascii (pyproject.toml) ... done
    Created wheel for aesthetic-ascii: filename=aesthetic_ascii-0.0.1-py3-none-any.whl size=24794090 sha256=577e4e55a7a4421afcedfe3878969fccfe893bcc6b1f130511ee2520d2398541
    Stored in directory: C:\Users\千江映月\AppData\Local\Temp\pip-ephem-wheel-cache-edc4tp4r\wheels\7a\23\70\43296f5f3ff4e796dc75a44512520842898bc580d909d344c3
  Successfully built aesthetic-ascii
  Installing collected packages: aesthetic-ascii
  Successfully installed aesthetic-ascii-0.0.1
  ```

  在编写完成package的代码后，可以使用上述方式先尝试本地安装，然后

- bulid （创建安装包）

  The build process creates a new directory `dist` which will contain a `.tar.gz` and `.whl` file — this is what we need to publish our package to PyPI.

  ```shell
  #  安装 bulid package
  pip install build
  
  # 创建安装包
  cd \Pycharm\textProcess\aesthetic_ascii
  
  python -m build
  
  # 创建完成后输出：
  Successfully built aesthetic_ascii-0.0.1.tar.gz and aesthetic_ascii-0.0.1-py3-none-any.whl
  
  ```

  本地安装创建的package

  ```shell
  cd \Pycharm\textProcess\aesthetic_ascii
  
  # 推荐使用
  pip install ./dist/aesthetic_ascii-0.0.1-py3-none-any.whl
  
  # 可以安装，不如第一种方式
  pip install ./dist/aesthetic_ascii-0.0.1.tar.gz
  
  
  
  # 或者
  tar -xzvf aesthetic_ascii-0.0.1.tar.gz
  cd aesthetic_ascii-0.0.1.tar.gz
  python setup.py install
  
  # 或者
  tar -xzvf aesthetic_ascii-0.0.1.tar.gz
  pip install aesthetic_ascii-0.0.1
  
  ```

  

- 发布安装包（Publish to TestPyPI，*TestPyPI* — a ‘test’ version of PyPI）

  首先发布到TestPyPI，如果没有出现异常情况，再发布到 PyPI

  ```shell
  # 第一步
  pip install twine
  
  # 第二步
  cd \Pycharm\textProcess\aesthetic_ascii
  
  python -m twine upload --repository testpypi dist/*
  
  # 第三步：测试安装
  pip install -i https://test.pypi.org/simple/ aesthetic-ascii
  
  # 测试发布无误后可以发布到PyPI
  python -m twine upload --repository pypi dist/*
  ```

  

- 