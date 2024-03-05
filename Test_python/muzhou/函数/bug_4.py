# coding:utf-8

# try……except……else……finally结构
# finally块无论是否发生异常都会被执行，能用来释放try块中申请的资源。


try:
    a = int(input('please input a target:'))
    b = int(input('please input a target:'))
    result = a / b
except BaseException as e:
    print('出错了', e)
else:
    print('计算结果为：', result)
finally:
    print('无论是否异常，都被执行的代码。')
print('程序结束')