names = ('高', '地')
ages = [1, 2, 3, 4]
jobs = {'a': 1}

for i, j, k in zip(names, ages, jobs):
    print(i, j, k)

# 不是用zip，也可以并行迭代多个序列。
for i in range(min(len(names), len(ages))):
    print('{}-{}'.format(names[i], ages[i]))

# 列表推导式
a = [x for x in range(1, 5)]
print(a)

b = [x * 2 for x in range(1, 5)]
print(b)

c = [x * 2 for x in range(1, 20) if x % 5 == 0]
print(c)

d = [x for x in 'abcdefg']
print(d)

e = [(row, col) for row, col in zip(range(1, 10), range(101, 110))]
print(e)

for cell in e:
    print(cell)

# 字典推导式
value = ['北京', '上海', '深圳']
citys = {city: id for id, city in zip(range(len(value)), value)}
print(citys)

citys = {id * 100: city for id, city in zip(range(1, len(value) + 1), value)}
print(citys)

# 利用字典统计字符出现的次数。
my_text = 'I love you, I love python, I love money'
dict_a = {}
for i in my_text:
    print('字符{}出现了{}次'.format(i, my_text.count(i)))
    # 上面这样会有很多重复字符。将字符和对应次数存到字典中，看是否是会解决重复这种情况
    dict_a[i] = my_text.count(i)
print(len(my_text))
print(dict_a, len(dict_a))
# 可以看到没有重复的，所以，再写的简洁写
dict_b = {i: my_text.count(i) for i in my_text}
print(dict_b)  # 其中i是key，my_text是value，因为字典中的key是不重复的。

print(dict_b == dict_a)  # True

# 生成器
iter_a = iter([1, 2, 3, 4])
for i in iter_a:
    print(i)

print('第二次输出生成器内容')
for i in iter_a:
    print('a')
    print(i)
# 为空了

# 小括号生成生成器
iter_b = (x for x in range(10))
print(iter_b)  # <generator object <genexpr> at 0x0000024523C5B258>
# 生成器可以使用for循环读取。
