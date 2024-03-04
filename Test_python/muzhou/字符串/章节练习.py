# coding:utf-8
# 车牌归属地

# # 统计字符串指定字符出现次数，不区分大小写。
# # 用到字符串中count方法，由于需求不区分大小写，可以将待检测的字符串转化为全部大写或者全部小写。
# s = 'HelloPython, helloJava, hellophp'
# word = input('请输入要统计的字符:')
# print('{0}在{1}中共出现了{2}次'.format(word, s, s.count(word)))

# 格式化输出商品的名称和单价
# 需求：使用列表存储一些商品数据，使用循环遍历输出商品信息，要求对商品的编号进行格式化为6位，单价保留2位小数，并在前面添加人民币符号输出

lst = [
    ['01', '电风扇', '美的', 500],
    ['02', '洗衣机', 'TCL', 1000],
    ['03', '微波炉', '老板', 400]
]

print('编号\t\t名称\t\t\t品牌\t\t单价')  # 注意，这里名称后面是跟了三个\t
for item in lst:
    for i in item:
        print(i, end='\t\t')
    print()  # 换行

# 格式化, 这一步使用的方法是赋值，重新给列表内容赋值

for item in lst:
    item[0] = '0000' + item[0]
    item[3] = '￥{:.2f}'.format(item[3])

# 重新遍历列表
print('编号\t\t\t名称\t\t\t品牌\t\t单价')
for item in lst:
    for i in item:
        print(i, end='\t\t')
    print()  # 换行


# 正则化，提取有效字符
#  在一串字符串中，提取出链接。其实就是找规律。
s = '''
DevTools failed to load source map: Could not load content for https://s1.hdslb.com/bfs/seed/log/report/950.ee096.function.chunk.js.map: HTTP error: status code 404, net::ERR_HTTP_RESPONSE_CODE_FAILURE
DevTools failed to load source map: Could not load content for https://s1.hdslb.com/bfs/seed/jinkela/short/b-mirror/biliMirror.umd.mini.js.map: HTTP error: status code 404, net::ERR_HTTP_RESPONSE_CODE_FAILURE
DevTools failed to load source map: Could not load content for https://s1.hdslb.com/bfs/static/jinkela/video/stardust-video.ad5ad3ef2cb1b8b45d70cdc0ba9ec0baae3066dc.js.map: HTTP error: status code 404, net::ERR_HTTP_RESPONSE_CODE_FAILURE
DevTools failed to load source map: Could not load content for https://s1.hdslb.com/bfs/seed/log/report/512.65972.function.chunk.js.map: HTTP error: status code 404, net::ERR_HTTP_RESPONSE_CODE_FAILURE
DevTools failed to load source map: Could not load content for https://s1.hdslb.com/bfs/seed/log/report/log-reporter.js.map: HTTP error: status code 404, net::ERR_HTTP_RESPONSE_CODE_FAILURE
'''
# 想提取到链接，找规律，发现都是https://s1.hdslb.com/bfs/开头，后面开始不一样。
