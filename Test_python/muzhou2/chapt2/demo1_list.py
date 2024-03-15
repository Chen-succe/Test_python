import random

a = [1, 3, 7, 3, 8]

a.sort()
print(a)

a.sort(reverse=True)
print(a)  # 降序

random.shuffle(a)
print(a)


# reversed内置函数

c = reversed(a)
print(c)
c_list = list(c)
print(c_list)
print(list(c))

print(sum(a), type(sum(a)))  # 22 <class 'int'>
print(max(a))
print(min(a))