a = 100


def f1():
    # 函数内可以使用全局变量a
    # print(a)
    b = 3
    c = 4
    global a
    a = 4
    print(a)
    print(locals())  # 打印输出局部变量
    print('*****' * 20)
    print(globals())  # 打印输出全局变量


f1()
# print(a)
# print(locals())  # 打印输出局部变量
# print('*****'*20)
# print(globals())  # 打印输出全局变量

import time

b = 1000


def test1():
    start_time = time.time()
    global b
    for i in range(1000000):
        b += i
    end = time.time()
    print('耗时{}'.format(end - start_time))


def test2():
    start_time = time.time()
    c = 1000
    for i in range(1000000):
        c += i
    end = time.time()
    print('耗时{}'.format(end - start_time))


test1()  # 0.05285763740539551
test2()  # 0.03490614891052246
