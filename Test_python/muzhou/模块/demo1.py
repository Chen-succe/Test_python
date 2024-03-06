# coding:utf-8

import math

print(type(math))  # <class 'module'>
print(id(math))
print(math.pi)  # 3.141592653589793
print(dir(math))

print(math.pow(2, 3))  # 8.0
print(math.ceil(0.9))
from math import pi

print(pi)
print(pow(2, 3))  # 8  这里pow很math.pow不一样

from math import pow

print(pow(2, 3))  # 这里使用的是导入后的pow。


