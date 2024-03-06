# coding:utf-8

class Student:
    def __init__(self, name, age):
        self.name = name
        self.__age = age

    def show(self):
        print(self.name, self.__age)


stu = Student('chen', 20)
stu.show()
# 在类外使用name和age
print(stu.name)
# print(stu.__age)  # AttributeError: 'Student' object has no attribute '__age'
# 实际上有__age属性，但是无法再类外访问。如果查看他所有的属性和方法呢
# dir(stu)， 使用dir（）方法
print(dir(stu))  # 可以找到_Student__age
print(stu._Student__age)  # 20  这样就可以访问了
# 在类外可以通过 _Student__age进行访问。那我们的封装还有什么意义？就是靠自觉性。
# 当你看到人家两个下划线定义的属性和方法你就别访问了，虽然可以通过特定代码访问，但还不不要去访问了。
