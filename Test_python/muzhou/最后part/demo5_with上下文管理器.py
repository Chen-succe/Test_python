# 自定义上下文管理器。

'''
MyContent实现了特殊方法__enter__(),__exit__()方法，称该类对象遵守 了上下文管理器协议
该类对象的实例对象，称为上下文管理器
'''


class MyContent(object):
    def __enter__(self):
        print('enter方法被执行了')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('exit方法被执行了')

    def show(self):
        print('show方法被执行了', 1/0)


myc = MyContent()
with myc as file:  # 等同于语句：with MyContent() as file:  因为MyContent()就是MyContent类的一个对象
    file.show()

