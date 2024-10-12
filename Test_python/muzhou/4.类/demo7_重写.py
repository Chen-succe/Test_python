# coding:utf-8

class Person(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def info(self):
        print('姓名{0}，年龄{1}'.format(self.name, self.age))


class Student(Person):
    def __init__(self, name, age, score):
        super().__init__(name, age)  # 这里需要传入父类需要的参数。super后面不要漏掉括号。
        self.score = score

    def info(self):
        super().info()
        print('分数 {}'.format(self.score))


class Teacher(Person):
    def __init__(self, name, age, year):
        super().__init__(name, age)
        self.year = year

    def info(self):
        super(Teacher, self).info()
        print('教龄 {}'.format(self.year))


# 测试
stu = Student('Jack', 20, '1001')
tch = Teacher('Jiang', 30, '2')
stu.info()  # 该方法是从Person类中继承的。
print('--------------')
tch.info()
