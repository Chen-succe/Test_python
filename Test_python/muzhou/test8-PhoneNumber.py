# coding:utf-8

# 创建一个集合，用来存放通讯录信息
set_PhoneNumber = set()
print(set_PhoneNumber)

# 录入5位好友的姓名和手机号
for i in range(1, 6):
    PhoneNumber = input(f'请输入第{i}位好友的姓名和手机号:')
    # 添加到集合中
    set_PhoneNumber.add(PhoneNumber)

for item in set_PhoneNumber:
    print(item)

# 知识点回顾，有：1. 空集合的创建，集合是无序的；2. 输出语句的三种写法；3. 集合添加数据的方法。



