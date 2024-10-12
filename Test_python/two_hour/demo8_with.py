# try语句使用
try:
    print('程序运行')
    # raise KeyError
except KeyError:
    print('key error 错误……')
else:
    print('程序未异常')
finally:
    print('程序无论是否出现异常都执行此语句')


# 上下文管理协议 - 魔法方法
class Sample:
    def __enter__(self):
        try:
            self.obj_file = open('file.txt')
        except FileNotFoundError:
            self.obj_file = None
        print('我被执行了:__enter__')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('我被执行了__exit__')
        if self.obj_file is None:
            print('当前文件不存在')
        else:
            self.obj_file.close()

    def run(self):
        print('程序启动')


# 定义了上面一个类，就可以使用with执行了
with Sample() as sample:
    sample.run()

import contextlib


@contextlib.contextmanager
def open_file(file_name):
    # file_obj = open(file_name)
    print('open:{}'.format(file_name))
    yield {'name': '顾安'}
    print('close:{}'.format(file_name))


# a = open_file('hi.txt')
# print(a)

with open_file('a.txt') as f:
    print('程序启动')
    print(f)  # 该语句可以拿到yield中的内容。
