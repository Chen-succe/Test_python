# coding:utf-8

class A:
    def __init__(self):
        self.a = 20
    pass


class B:
    pass


class C(A, B):
    def __init__(self, name, age):
        super().__init__()  # 总忘记继承super这行代码。
        self.name = name
        self.age = age

    # def __str__(self):
    #     return 'I love you'
    def __add__(self, other):
        return self.age + other.age

    def __len__(self):
        return len(self.name)


x = C('Jack', 30)

print(x.__dict__)  # 实例对象的属性字典
print(C.__dict__)
print(x.__class__)  # 输出了对象所属的类
print(C.__bases__)  # 输出C的父类
print(C.__base__)  # 输入跟他最近的父类
print(C.__mro__)  # 查看类的层次结构
print(A.__subclasses__())  # 查看子类

c1 = C('Jack', 20)
c2 = C('Chenchen', 28)
print(c1 + c2)
print(len(c1))
print(id(object))
print(id(object))