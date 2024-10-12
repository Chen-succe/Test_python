a = 3
b = 'love you'
c = 5
d = 'I'
print(type(a))
print(type(b))
print(id(a))
print(id(b))
print(id(c))
print(id(d))
# print(help())

a, b = 1, 2
print(a, b)
a, b = b, a
print(a, b)
print(3.14e2)
#  打印九九乘法表

for m in range(1,10):
    for n in range(1, m+1):
        print('{}*{}={}'.format(m, n, (m*n)), end='\t')
    print()

for i in range(10):
    if i > 5:
        break
    print('{}比5小'.format(i))
else:
    print('end')


for i in range(10):
    if i > 5:
        continue
    print('{}比5小'.format(i))
else:
    print('end')