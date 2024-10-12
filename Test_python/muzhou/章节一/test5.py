# 集合和元祖
# a = tuple()
# a1 = (1, 2)
# print(a1.index(1))
# print(a1.count(1))

# #  集合
# a = set()
# aa = {1}
# print(type(aa))

# dict

# a = str()
# print(a.index(''))
# print(a + 'qqq')
# a = dict()
# print(len(a))
# b = (['a',[1,2,3],], [1,2], [(1,2),'s'] )
# print(dict(b))
# print(a)

# a = {'name': 'chen', 'age': 18, 'major': 'computer', 'age':16, 'age':20}
# # print(a)
# # print(a.items())
# # print(a.keys())
# # print(a.values())
# # print(a.get('name'))
# # print(a.get('major', 'English'))
# # print(a.get('weight', '50'))
# aa = a.values()
# # print(aa, type(aa))
# # print(aa[0], type(aa[0]))
# for i in aa:
#     print(i)
#     print(type(i))
#
# result = a.popitem()
# print(result)
# print(result[0])

# in // not in

a = {1, 2, 3, (1, 2, 3), 'se'}
print(a)
print('se' in a)
print(1 in {1, '2'})

a = {1, 2, 3, (1, 2, 3), 'se'}
b = a
print(a is b.copy())  # False

c = 'a'
d = 'a'
print(c is d)  # True
print(c == d)  # True
print(a == b)  # True
print(a == b.copy())  # True

s1 = 'a'
s2 = 'a'
print(s1 is s2)  # True
l1 = []
l2 = []
print(l1 is l2)  # False
l1 = [1]
l2 = [1]
print(l1 is l2)  # False

d1 = {}
d2 = {}
print(d1 is d2)  # False

set1 = set()
set2 = set()
print(set1 is set2)  # False

tuple1 = tuple()
tuple2 = tuple()
print(tuple1 is tuple2)  # True

tuple1 = (1,)
tuple2 = (1,)
print(tuple1 is tuple2)  # True

print(type(str(d1)), str(tuple1) + str(set1) + 'asf' + str([12, 3, 4]))
print('sfsfd')
print(type(True), type(3.23))  # <class 'bool'>  <class 'float'>
a = 1
b = str(a)
print(type(a), type(b))
print('sdf' + str(True))
c = str([1, 2, 3])
print(len(c), c[0])

print(int('123'))
print(float('123'))
print(float(False))
print(int(False))
print(float('-123.3'))  # -123.3
print(list({1,2,3,'123', 12, '1', 'a', (1,2,3)}))
a = {1,2,3,'123', 12, '1', 'a', (1,2,3)}
list_a = [1,2,3]
b = set(list_a)
print(b)
a.update(b)
print(a)
