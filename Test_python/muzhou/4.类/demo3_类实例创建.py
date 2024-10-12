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
        print(self.native_pace)

    # 静态方法，静态方法中是不能写self的。
    @staticmethod
    def mm():
        # print(native_pace)  # 无法访问，会报错
        print('我是静态方法')

    # 类方法，传的参数叫cls
    @classmethod
    def cm(cls):
        # print(native_pace)  # 无法访问，会报错。用错方式了，实例方法和类方法都可以访问类属性，要用self和cls调用
        print(cls.native_pace)
        print('我是类方法')


# 在类之外定义的称之为函数，类内定义的成为方法。
def eat():
    print('类外的学生在吃饭')


stu1 = Student('张三', 18)
stu2 = Student('李四', 10)
print(stu1)
print(id(stu1))  # 1248364033032
print(type(stu1))  # <__main__.Student object at 0x00000122A84B7408>
# 1248364033032的十六进制数等于122A84B7408
print(id(Student))

stu1.eat()  # 对象名.方法名()
print(stu1.name)
Student.eat(stu1)  # 第40行与38行功能一样，都是调用Student中的eat方法。
# 类名.方法名(类的对象)-->实际上就是方法定义处的self
print(stu1.native_pace)
# stu1.native_pace = 'tianjin'  # 实例对象修改类属性无效
Student.native_pace = 'tianjin'
print(Student.native_pace)

print('---------类方法的使用-------------')
Student.cm()

print('---------静态方法的使用-------------')
Student.mm()
print('---------实例调用类方法-----------')
stu1.cm()
stu1.mm()
stu2.cm()