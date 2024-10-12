# coding:utf-8

# try……except……else结构
# 如果try快中没有抛出异常，则执行else快，如果try中抛出异常，则执行except块。
# 如果不知道会出现什么异常，则使用BaseException，会打印出所有异常，如果没有异常出现，则执行else。

try:
    a = int(input('please input a target:'))
    b = int(input('please input a target:'))
    result = a / b
except BaseException as e:
    print('出错了', e)
else:
    print('计算结果为：', result)




