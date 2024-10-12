# coding:utf-8
# 类的创建

class Student:
    native_pace = 'qingdao'  # 直接写在类中的变量，称为类属性。

    def __init__(self, name, age):
        self.name = name  # self.name是实例属性，进行了一个赋值操作，将局部变量name的值赋给实例属性。
        self.age = age

    # 实例方法
    def eat(self):
        print('学生在吃饭')

    # 静态方法，静态方法中是不能写self的。
    @staticmethod
    def mm():
        print('我是静态方法')

    # 类方法，传的参数叫cls
    @classmethod
    def cm(cls):
        print('我是类方法')


# 在类之外定义的称之为函数，类内定义的成为方法。
def eat():
    print('类外的学生在吃饭')


eat()
a = Student()
a.eat()
