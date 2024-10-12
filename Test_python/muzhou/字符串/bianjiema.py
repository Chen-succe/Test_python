a = '你好中国'
gbk_str = a.encode('gbk')
print(gbk_str)
utf_str = a.encode()
print(utf_str)

print(gbk_str.decode('gbk'))
print(bytes.decode(gbk_str, 'utf-8', errors='replace'))

print('12324324553'.isdigit())
print('一二三'.isdigit())
print('0b010101'.isdigit())
print('一二三'.isalnum())
print('0b010101'.isalnum())
print('一二三'.isalnum())
print('sjiag'.isalnum())
print('sfd'.isnumeric())
print('一二三'.isnumeric())
print('1234jskdhg你好'.islower())  # True

print('**' * 8)
# 首字符大写
print('Hello'.istitle())  # True
print('HelloWorld'.istitle())  # False
print('Helloworld'.istitle())  # True
print('Hello world'.istitle())  # False
print('Hello World'.istitle())  # True

print('HelloWorld'.title())  # Helloworld


