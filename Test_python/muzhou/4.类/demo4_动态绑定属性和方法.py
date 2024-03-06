# coding:utf-8

# Python动态绑定属性和方法
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def eat(self):
        print(self.name + '在吃放')


def show():
    print('我是一个show方法')


stu1 = Student('张三', 20)
stu2 = Student('李四', 20)
print(id(stu1))
print(id(stu2))
print('-------------为stu1动态绑定性别属性----------')
stu1.gender = '女'
print(stu1.name, stu1.age, stu1.gender)
print(stu2.name, stu2.age)

print('---------------为stu1动态绑定show方法')
stu1.show = show
stu1.show()
# stu2.show()  # 报错，因为stu2没有绑定show方法
