# coding:utf-8
import re

# 有些东西是不能在评论中出现的的，比如在广告语中是不能写什么什么第一，什么什么最好的。
# 比如黑客是不能出现在评论中的。这里只是举个例子，还有很多情况，在有些视频下有些特定词汇不能出现
# 或者有些违纪的词汇不能出现在评论中。
pattern = '黑客|破解|反爬'  # 比如有这种模式字符串是不能出现在评论区的。
s = '我想学Python，想破解一些VIP视频，Python可以实现反爬吗'  # 比如这是要替换的字符串。

# 现在，要实现的功能就是把字符串中有违纪的即pattern中的字符，替换为xxx，
new_s = re.sub(pattern, 'xxx', s)
print(new_s)  # 我想学Python，想xxx一些VIP视频，Python可以实现xxx吗
# 最后效果就是将违禁字符替换为xxx


# 网址中有split的
# 比如分割百度浏览器的地址，百度的地址有固定规则，?&等连接不同内容
s2 = 'https://www.baidu/s?wd=ysj&ir=utf-8&tn=baidu'
pattern2 = '[?|&]'  # 这个pattern表示中括号中的两个字符都可以匹配
# 注意：该方法当有多个切割符如这里两个，则需要用[]括起来。如果只有一个，pattern = '?' 这样写即可
lst = re.split(pattern2, s2, 2)
print(lst)