from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name="monotonic_align",
    ext_modules=cythonize(r"C:\Users\LSY\Desktop\大创2024\基于人工智能的虚拟主播的设计与实现\Code\vits_chinese-bert_vits\monotonic_align\core.pyx"),
    include_dirs=[numpy.get_include()],
)
