# 函数作为对象，一种语法糖

def outer(func_obj):
    def inner():
        print('这是一个装饰器')
        func_obj()

    return inner  # 外层函数返回是内层函数对象/引用


def mid():
    print('这是测试函数')


inners = outer(mid)
inners()


# 语法糖方式
def outer(func_obj):
    def inner():
        print('这是一个装饰器')
        func_obj()

    return inner  # 外层函数返回是内层函数对象/引用

@outer
def mid():
    print('这是测试函数')


mid()
