a = tuple()
b = (1,2,3)
c = 1,
print(type(c))

d = tuple(range(10))
e = tuple('abc')
print(d)
print(e)


# 推导式得到的是一个生成器

aa = (x*10 for x in range(10))
print(aa)  # <generator object <genexpr> at 0x00000139FDF0E6D0>
# cc = tuple(aa)
ccc = list(aa)
# print(cc)
print(ccc)  # [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
# 转列表转元祖都可以，只不过都是只能转一次。

# next方法测试
aa = (x*10 for x in range(3))
print(aa.__next__())  # 0
print(aa.__next__())  # 10
print(aa.__next__())  # 20
# print(aa.__next__())  # 到这里报错了

aa = (x*10 for x in range(3))
a1 = next(aa)
print(a1)  # 0
print(next(aa))
print(next(aa))
# print(next(aa))  # 报错

# 上面是生成器，还有一个迭代器
l = [1,2,3,4]
print(l)
l_iter = iter(l)
print(l_iter)  # <list_iterator object at 0x000002B059B4A940>
for i in l_iter:
    print(i)

aa = (x*10 for x in range(3))
for i in aa:
    print(i)

# 生成器，迭代器都可以使用for循环访问
