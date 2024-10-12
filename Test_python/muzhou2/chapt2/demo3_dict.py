# 创建字典的几种方法

a = {'name': 'chen', 'age': 18}
b = dict(name='chen', age=18)
print(id(a), id(b))
print(a)
print(b)
c = dict([('name', 'chen'), ('age', 18)])  # 元祖放到列表中
print(c)

# 通过zip创建
k = ['name', 'age']
v = ['chen', 18]
d = dict(zip(k, v))
print(d)
e = dict(list(zip(k, v)))
print(e)
# 通过fromkeys
f = dict.fromkeys(['name', 'age'])
print(f)

print(a['name'])
print(a.get('name'))
print(a.get('wo', '不存在'))

print(a.items())
print(a.keys())
print(a.values())
print(len(a))
print('name' in a)
b = {'name': 'jiang'}
a.update(b)
print(a)
print(b)
b.clear()
print(b)
age = a.pop('name')
print(age)
print(a)
a_item = a.popitem()  # 随机移出，没有顺序。这个方法可以将字典中元素一个一个移出。
print(a)
print(a_item)

# 序列解包
print(c)
name, age = c.items()
print(name[0])  # name
print(age)  # ('age', 18)
name, age = c.values()
print(name, age)

# 表格数据使用字典和列表 存储和访问
r1 = {'name': 'a', 'age': 18, 'salary': 9000, 'city': '北京'}
r2 = {'name': 'b', 'age': 20, 'salary': 8000, 'city': '上海'}
r3 = {'name': 'c', 'age': 19, 'salary': 7000, 'city': '广州'}
tb = [r1, r2, r3]
print(tb)
print(tb[1].get('salary'))
for i in range(len(tb)):
    print(tb[i].get('name'), tb[i].get('age'), tb[i].get('salary'), tb[i].get('city'))
