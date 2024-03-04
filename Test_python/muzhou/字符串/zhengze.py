# coding:utf-8
# 导入
import re

pattern = '\d\.\d+'  # 第一个\d是匹配整数，第二个转义字符加.就是.的意思，(即匹配所有数字.数字形式，如2.3)
# 第三个\d后面加一个+号限定符表示限定次数，可以多次

s = 'I study python 3.10 every day, 0.5'  # 待匹配的字符串

match = re.match(pattern, s, re.I)  # re.I 表示不区分大小写 I表示ignore，但是这
# 里匹配pattern中只匹配了整数，整数是没有大小写的，实际上这里的match方法中第三个参数不起作用。

print(match)  # None
#  返回结果为None，即不符合匹配规则。因为match是从头匹配，这表明s开头没有符合pattern格式的字符。

s2 = '3.10python I study every day'
match2 = re.match(pattern, s2, re.I)
print(match2)  # 结果显示匹配成功，并且有详细的描述。
#  <re.Match object; span=(0, 4), match='3.10'>  span表示从哪个位置，match表示匹配到的字符
# 查看match中的详细内容，可以通过使用match中的方法查看。
print('匹配值的起始位置：', match2.start())
print('匹配值的结束位置：', match2.end())
print('匹配值的位置元祖：', match2.span())
print('待匹配的字符串：', match2.string)  # 注意，这个不是方法
print('匹配的数据：', match2.group())

search = re.search(pattern, s, re.I)
print(search.group())

findall = re.findall(pattern, s, re.I)
print(findall)