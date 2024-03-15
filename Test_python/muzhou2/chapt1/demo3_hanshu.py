def add(a, b, c):
    sum = a + b + c
    return sum


result = add(1, 2, 3)
print(result)

print(add)  # <function add at 0x0000017B47E82EA0>
print(id(add))  # 1628999003808
print(type(add))  # <class 'function'>

a = 1
print(a)  # 1
print(id(a))  # 1930801344
print(type(a))  # <class 'int'>

# print(id(add()))

# def addd():
#     return 3
#
#
# print('-----')
# print(id(addd()))  # 1930801408
# print(id(3))  # 1930801408
