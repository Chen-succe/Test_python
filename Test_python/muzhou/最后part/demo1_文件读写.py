# coding:utf-8
file = open('a.txt', 'w', encoding='utf-8')  # 这里，使用utf-8和gbk，打开a.txt都是正常的，
# 但使用notpadd++打开时，utf-8正常，gbk的乱码，而使用记事本打开，两种写入都是正常打开的。
file.write('中国')
file.close()

file2 = open('b.txt', '+w', encoding='utf-8')
# content = file2.readline()
# read = file2.read()
file2.writelines(['1,2,3', 'sjjdk', 'java', 'python'])
file2.flush()
file2.seek(0)
print(file2.read())
file2.seek(0)
content_ = file2.readlines()
for i in content_:
    print(i)
file2.close()

print('----------------------')
file3 = open('b.txt', 'r', encoding='utf-8')
contents = file3.readlines()
print(contents)
for i in contents:
    print(i)
file3.close()

# 写入文件
file4 = open('c.txt', 'w')
contents_lt = ['123', 'jave', 'python']
for i in contents_lt:
    file4.write(i + '\n')
file4.close()

# 读出文件
file5 = open('c.txt', 'r')
contents = file5.readlines()
for i in contents:
    print(i.strip())  # 去掉空格行
file5.close()