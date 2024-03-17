class A:
    def __init__(self):
        print('A')


class B(A):
    def __init__(self):
        print('B')
        # A.__init__(self)
        # super(B, self).__init__()
        super().__init__()


b = B()
# print(b)

# 需要创建一个类，让我们自定义的类具有多线程执行的特征
import threading


# 可以重用线程类Tread中定义的属性（也就是__init__方法），如何重用呢，我们把self.thread_name注释掉
# 使用super调用父类Thread的init方法，将thread_name传给name参数，就实现了代码的重用。
class MyThread(threading.Thread):
    def __init__(self, thread_name, user):
        self.user = user
        # self.thread_name = thread_name
        super().__init__(name=thread_name)
        # super(MyThread, self).__init__(name=thread_name)



# 问题3：super函数的运行过程
class D:
    def __init__(self):
        print('d')


class C(D):
    def __init__(self):
        print('c')
        super().__init__()


class BB(D):
    def __init__(self):
        print('b')
        super(BB, self).__init__()


class A(BB, C):
    def __init__(self):
        print('a')
        super().__init__()


print('------------')
a = A()
print(A.__mro__)