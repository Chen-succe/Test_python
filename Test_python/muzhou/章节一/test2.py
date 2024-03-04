# name = '小仙女长相甜美并且温柔'
# print(name[1:])
# print(name[-1])
# print(name[1:-1])
# print(name[-1:-3])
# print(name[-3:-1])
# print(name[::-2])
#
# b = 'a'
# b= ord(b)
# print(b)

# A = 's'
# B = "s"
# C = '''s'''
# print(A==B==C) # True
# print(id(A), id(B), id(C))
#
# a = 18
# b = 'beijing'
# c = '我现在居住在{1}，今年{0}岁,{2}'.format(a, b, "welcome")
# print(c)
# result = b.find('iji', )
# print(result) # 4
# result = b.count('ij', )
# print(result)

#
# b = 'beijing'
# result = b.replace('i', 'o', 1)
# print(result) # beojing


b = '  aaaadaafdsfaaa  '
result = b.strip()  # aaaadaafdsfaaa
result = result.strip('a')  # daafdsf
print(result)
print('I am %s years old, %d, %s， %.3f' % (12.3, 23.3, [123], 22.33))

# b = 'beijing beijing'
# result = b.split('i', 2)
# print(result)  # ['be', 'j', 'ng beijing']
print(F'{b}')

print('我今年赚了{1:.3%}元'.format(1000000.2345, 2000000.2345))
print("eat {}".format([1, 2, 3, 4]))
# print(1.1 + True)
# b = 1.1 + True
# print(type(b))
# print(True + True)
# a = True + True
# print(type(a))
# c = True and True
# print(a + c)
# d = 'adgdasg'
# print(d[-6:-1])
# print(d[-6:-1:2])
# print(d[-6:])
# print(d[-6::2])

list1 = [2, 2, 3]
print(list1.index(2))
list2 = [2, 3, 4, [23, 4]]
lists = list1 + list2
print(lists)
list_x = list2 * 2
print(list_x)
print(list_x[-7: 7])
print(list_x[::-1])
del list_x[0]
print(list_x)
# list_x.remove('2')
list_x.pop(0)
print(list_x.reverse())
print(list_x)

list_sort = ['4', '我', 'hi', 'word', '2']
list_sort.sort()
print(list_sort)
print(ord('我'))
print(ord('1'))