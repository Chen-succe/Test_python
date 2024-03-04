# Python条件语句

# a = input('请输入你的成绩：')
# a = int(a)
# if a >= 85:
#     print('perfect')
# elif a >= 60:
#     print('just')
# else:
#     print('not pass')
#
# d = {'a': 1}
# for i in d:
#     print(i)
#
# for i in d.keys():
#     print(i)
#
# for i in d.values():
#     print(i)

# 函数
def show():
    print('there is no return ')
    return


show()
print(show())


def fe_123():
    pass


def add_number(a, b='123'):
    print(b + str(a))


add_number(2)


def abc(a, b=None):
    if b is None:
        b = []
    b.append(a)
    print(b)


abc(100)
b = [1, 2, 3, 4, 5]
print(b)
abc(200)

print('-------')

c = [1, 2, 3, 4]
b = [1, 2, 3, 4, 5]


def abcd(a, b=[]):
    b.append(a)
    c.append(a)
    print(b)


abcd(100)
abcd(200)
print(c)
print(b)


def abc(a=100, b=200, c=300):
    global e
    e = 10
    print(a + b + c)


def abd():
    e = 100
    print(e)


abc(200, 200, c=200)
a = ()
print(a, type(a))


def abc(a, **kwargs):
    print(a)
    print(kwargs)


r = 'fdf'
d = {'abf': 18, }
abc(20, **d)


def bac(a, *args):
    print(a)
    a1, a2, a3 = args
    print(a1,a2,a3)
    print(args)

bac(30, *[2,3,4])

def vda(a, b,c,d):
    print(a)
    print(b)
    print(c)
    print(d)

vda(*[1,2,3,], 1)
