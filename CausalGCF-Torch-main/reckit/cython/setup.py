'''# cython: language_level=3'''
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("eval_matrix.pyx"),  # 分别换上不同的pyx文件就可以生成对应的pyd文件，在此文件目录的reckit/cython中，然后复制到外层的cython文件中
    include_dirs=[np.get_include()]
)
# setup(
#     random_choice = cythonize("random_choice.pyx")
# )
# setup(
#     tools = cythonize("tools.pyx")
# )