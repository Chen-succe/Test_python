#  解包

dict_ = {'名字': 'xiaoming', 'bb': 18, 'cc': 49}


def abc(名字, bb, cc):
    print(名字)
    print(bb)
    print(cc)


abc(**dict_)


def abd(a, *d):
    print(a)
    print(d)


abd(1, *(1, 2, 3))
abd(1, 1, 2, 3, 4, 5, 6, 7)


def abb(a, **d):
    print(a)
    print(d)


print('*' * 8)
abd(1, (1, 2, 3))

abb(1, x=1, y=2)
abb(1, **{'x': 1, 'y': 2, 'z': 3})

print('-' * 20)


def acc(a, d='18', *b, **c):
    print(a)
    print(b)
    print(d)
    print(c)


acc(*(1, 2, 3, [1, 2, 3]), **{'c': 2})

# return 关键字
print('-------' * 4)


def abc(a, b, c):
    return str(a) + 'love', str(b) + 'you', c


result = abc(3, 4, '.')
print(result, type(result))

print('-----' * 9)


# 函数嵌套

def abcde():
    def cbdsd():
        return [1, 3, 4]

    return cbdsd


result = abcde()
print(result())


# 千年虫.需求：2位整数的在年份前面加19，年份是00，年份前面敬爱200
lst = [88, 89, 90, 00, 99]
lst_new = []
for i in range(len(lst)):
    compare = lst[i]
    print(compare)
    if str(compare) == '0':
        j = '200' + str(compare)
        lst_new.append(j)
    else:
        j = '19' + str(compare)
        lst_new.append(j)

print(lst)
print(lst_new)

for i in ([1,2,3,0,000,0000]):
    print(i)

print(0000)
a = ()
print(type(a))
print("#"*20)
# print(max(10))
print(max((10,)))
b = {}
c = 'a'