s1 = 'hello worLd'
new_s = s1.upper()
new_s2 = s1.lower()
print(s1, new_s, new_s2)

lst = s1.split(' ')
print(lst)

# 注意，空格也是字符串

# 统计子串在指定字符串中出现的次数。
print(s1.count(' ', 0, 3))

# 检查操作，首次出现的位置
print(s1.find(' '))
print(s1.find('f'))  # 没找到的，返回-1

# o首次出现的位置
print(s1.index(' '))
# print(s1.index('f'))  # 没找到会报错 ValueError: substring not found

# index和find区别就是查找的时候，没找到find返回-1，而index会报错。

# 判读前缀和后缀，返回值为bool类型
print(s1.startswith('h'))
print(s1.endswith('r'))
print('python.py'.endswith('.py'))  # 判断一个文件是否是Python文件
print('text.txt'.endswith('.txt'))  # 判断一个文件是否是txt文件。同理，可以延伸到其他文件类型
'''
True
False
True
True
'''

print(s1)
print(s1.center(30, '1'))  # 111111111hello worLd1111111111
print('/'.join(['1', '2', '3']))  # 1/2/3

s2 = 'ohello, hello, heoll, hleleo'
print(s2.strip('lo'))
age = 19
score = 100.34
print(f"{age}")
print('score is %.1f' % (score))

# （3） 使用字符串的format方法
print('name:{0:*^8}, age:{1:$<8}, score:{2:&>8}'.format('duo', age, score))

# 千位分隔符
scores = 100000000000
print('score is {0:,}, {0:b}'.format(scores))
