a = 'hello'
b = 'world'

print(a + b)

a = [1, 3]
b = [2, 5]
print(a + b)
print('hello''world')  # helloworld

str_s = 'ajidohgaioslnkdlgng'
str_new = ''
for i in str_s:
    if i not in str_new:
        str_new += i
print(str_new)  # ajidohgslnk

# 集合+列表排序
str_s = 'ajidohgaioslnkdlgng'
str_new = set(str_s)
print(str_new)
list_str = list(str_new)
print(list_str)
list_str.sort(key=str_s.index)
print(''.join(list_str))  # ajidohgslnk
list_str.sort(key=ord)
print(list_str)
print(''.join(list_str))  # adghijklnos

# 列表去重
a = ['金星', '木星', '木星', '金星', '水星', '金星']
b = set(a)
print(b)
list_b = list(b)
print(list_b)
list_b.sort(key=a.index)
print(list_b)