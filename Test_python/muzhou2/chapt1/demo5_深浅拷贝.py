import copy

a = 3
b = [10]
a_copy = copy.copy(a)
b_copy = copy.copy(b)
print(id(a), id(b), end='    ')
print()
print(id(a_copy), id(b_copy), end='    ')

a_deepcopy = copy.deepcopy(a)
b_deepcopy = copy.deepcopy(b)
print()
print(id(a_deepcopy), id(b_deepcopy), end='    ')
b.append(20)
print(id(b))

# 传递不可变对象，包含子对象是可变的
a = (10, 20, [5, 6])
print('a:', id(a))


def test1(m):
    print('m:', id(m))
    m[2][0] = 888
    print(m)
    print('m:', id(m))


test1(a)
print(a)


def f1(a, b, *c):
    print(a, b, *c)  # 8 9 19 20
    print('c:', c)  # c: (19, 20)


f1(8, 9, 19, 20)


def f2(a, b, **c):
    print(a, b, c)


f2(8, 9, name='good', age='18')
print('------------------------')
a = 2
b = 3
dict1 = dict(a=200, b=100)
print(dict1)
d = eval('a+b', dict1)
print(d)