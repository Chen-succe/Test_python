# os操作

import os
# os.system('notepad.exe')  # 打开记事本
# os.system('calc.exe')  # 打开计算器
# os.startfile('C:\Program Files (x86)\Tencent\WeChat\WeChat.exe')  # 启动程序，比如微信，

# os模块除了可以调用操作系统，还可以对文件和目录进行操作
print(os.getcwd())  # 返回当前工作目录
print(os.listdir('../最后part'))  # 返回指定路径下的文件和目录信息。


os.mkdir('newdir2')  # 创建目录
os.makedirs('a/c/d', exist_ok=True)  # 创建多级目录
os.rmdir('newdir2')  # 删除目录
os.removedirs('a/c/d')  # 删除多级目录
os.chdir('D:/D/PythonFile/2024_test/Test_python/muzhou') # 将这个目录设置为当前目录，再调用os.getcwd看看结果
print(os.getcwd())
print(os.listdir(os.getcwd()))
