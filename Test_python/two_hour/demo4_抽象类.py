# 开发一个web框架，在框架中实现缓存功能
from abc import ABC, abstractmethod
from collections.abc import Sized


class Student:
    def __len__(self):
        return 0


stu = Student()
print(isinstance(stu, Sized))  # True 因为实现了len方法
print(hasattr(stu, '__len__'))  # True


# 如果想让用户必须实现 get_cache  set_cache这两个方法，就可以在下面添加装饰器
class Cache(ABC):
    @abstractmethod
    def get_cache(self):
        pass

    @abstractmethod
    def set_cache(self, value):
        pass


class TulingOnline(Cache):
    def get_cache(self):
        pass

    def set_cache(self, value):
        pass


tul = TulingOnline()
print(tul)


