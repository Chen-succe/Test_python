# 遍历对象

class Student:
    def __init__(self, student_list):
        self.stu_list = student_list

    # 使用魔法方法 __getitem__
    def __getitem__(self, item):
        return self.stu_list[item]

    def __len__(self):

        return len(self.stu_list)

    def __str__(self):
        return '这是我自己定义的类，'


stu = Student(['张三', '王五', '李四'])

# for i in stu.stu_list:
#     print(i)

for i in stu:
    print(i)

print(stu[0])
print(len(stu))
print(stu)
list_a = ['hi']
list_a.extend(stu)
print(list_a)
list_a.extend((1,2,3))
print(list_a)