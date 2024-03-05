# coding:utf-8
# 输入两个整数，并进行除法
# a = input('please input a target:')
# b = input('please input a target:')
# print('result is {}'.format(int(a)/int(b)))


# 修改, 使用多个except，捕获多个错误
try:
    a = input('please input a target:')
    b = input('please input a target:')
    print('result is {}'.format(int(a) / int(b)))
except ZeroDivisionError:  # 捕获除数为0异常
    print('除数不能为0！')
except ValueError:
    print('只能输入数字串')  # 捕获输入为空异常，以及输入的不是数值的异常

print('程序结束')


# 修改， 使用BaseException，捕获所有错误，并且可以打印，效果同except后面不写，
# 不写则无法打印出e，即无法看到具体异常是什么。
try:
    a = input('please input a target:')
    b = input('please input a target:')
    print('result is {}'.format(int(a) / int(b)))
# except ZeroDivisionError:  # 捕获除数为0异常
#     print('除数不能为0！')
# except ValueError:
#     print('只能输入数字串')  # 捕获输入为空异常，以及输入的不是数值的异常

except BaseException as e:
    print(e)
    print('wrong')
print('程序结束')
