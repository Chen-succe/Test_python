class Student:
    def __str__(self):
        return 'I love you'


stu = Student()

print(dir(stu))
print(stu)  # <__main__.Student object at 0x00000234F9250E48>,这是父类object中的__str__方法起的作用，
# 我们可以重写该方法，让其输出我们想让他输出的内容。
