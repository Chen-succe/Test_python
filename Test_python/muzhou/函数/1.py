# coding:utf-8
# 计算1到多少的和

def get_sum(num):
    s = 0
    for i in range(1, num + 1):
        s += i
    print(f'1到{num}之间的和为:{s}')


# 一个函数只有定义没有调用是不会执行的，只有被调用才会执行。

get_sum(10)
get_sum(100)
get_sum(1000)


# 体会到函数调用的好处了吗


def happy_birthday(name='chen', age=19):
    print('祝{0}生日快乐'.format(name))
    print(str(age) + '岁生日快乐')


happy_birthday(29, 'chen')
happy_birthday(age=29, name='chen')
happy_birthday(name='chen', age=29)
happy_birthday('chen')
happy_birthday(29)

# 可变参数
a = 1, 2, 3, 4
print(type(a))  # <class 'tuple'>
print(a)  # (1, 2, 3, 4)


def print_(*item):
    print(type(item))
    print(item)
    for i in item:
        print(i)


print_(1, 2, 3, 4)  # 会自动封装成元祖
print_([1, 2, 3, 4])  # 会变成元祖的一个元素
print(*[1, 2, 3, 4])  # 如果想一个一个取，需要解包*。


def fun2(**kwargs):
    print(type(kwargs))
    print(kwargs)
    for k, v in kwargs.items():
        print(k)
        print(v)
        print('------')


fun2()
fun2(name='chen', age=29, wight=50)
fun2(**{'name': 'chcen', 'age': 18, 'weight': 50})
