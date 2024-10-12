class A:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def add_attribute(self, salary):
        self.salary = salary
        print('增加属性')

    def print_all(self):
        print('打印所有属性：', self.name, self.age, self.salary)


a = A('chen', 28)
a.add_attribute('70w')  # 在init外增加的属性，如果想使用，需要先将调用增加该属性的函数。
a.print_all()
b = A('chenchen', 28)
a.sex = '女'  # 只有对象b有sex这个属性。

c = A('c', 20)
A.add_attribute(c, '75w')
A.print_all(c)


# print(dir(A))
# print(dir(a))
# print(a.__dict__)  # {'name': 'chen', 'age': 28, 'salary': '70w', 'sex': '女'}
# print(type(A))  # <class 'type'>
# print(type(a))  # <class '__main__.A'>
# print(isinstance(a, A))  # True

class Student():
    company = '学堂'
    count = 0

    def __init__(self, name, age):
        self.name = name
        self.age = age
        Student.count = Student.count + 1

    @classmethod
    def lei_method(cls):
        print(cls.count)


s1 = Student('a', 1)
print(Student.count)  # 1
s2 = Student('b', 2)
print(Student.count)  # 2
Student.lei_method()  # 2


# 析构函数，python自动垃圾回收机制

class Person:
    def __init__(self):
        self.name = 'c'

    def show(self):
        print('显示{}'.format(self.name))

    def __del__(self):
        print('销毁对象{}'.format(self))


p1 = Person()
p2 = Person()
p1.show()


def work(s, a):
    print('好好赚钱')

Person.work = work
p1.work('a')

