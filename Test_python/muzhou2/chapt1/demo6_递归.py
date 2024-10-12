def f1(n):
    print('Start:', n)
    if n == 1:
        print('recursion over!')
    else:
        f1(n - 1)
    print('end:', n)


f1(5)


# 阶乘

def factorical(n):
    if n == 1:
        return 1
    else:
        return n * factorical(n - 1)


print(factorical(4))

# 内部函数
# 测试nonlocal， global关键字用法
a = 100


def outer():
    b = 10

    def inner():
        # nonlocal b
        # print('inner b:', b)
        b = 20
        print(b)

        global a
        a = 1000

    inner()
    print('outer b:', b)


outer()
print(a)