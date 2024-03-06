# coding:utf-8
# 类的创建

class Student:  # Student 为类的名称（类名），由一个或多个单词组成，每个单词的首字母大写，其余小写。
    # 不遵循这个规范也可以，但会导致代码很丑。这个规范是程序员默认的。
    pass


# Python中一切皆对象。Student是对象吗？内存有开空间吗？

print(id(Student))  # 2621826290904
print(type(Student))  # <class 'type'>
print(Student)  # <class '__main__.Student'>
