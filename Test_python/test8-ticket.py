# coding:utf-8

# 创建字典，用于存储车票的信息，使用车次做key，使用其他信息做value
dict_ticket = {
    'G1569': ['北京南-天津南', '18:06', '18:39', '00:33'],
    'G1567': ['北京南-天津南', '18:15', '18:49', '00:34'],
    'G8917': ['北京南-天津南', '18:20', '19:19', '00:59'],
    'G203': ['北京南-天津南', '18:35', '19:09', '00:34'],
}

# 遍历字典中的元素
print('车次   出发地-到达地 出发时间    到达时间隔   历时时长')
for key in dict_ticket.keys():
    print(key, end=' ')  # 为什么不换行，因为车次和车次信息在一行输出。end表示以什么结尾。否则默认以换行结尾。
    # 遍历车次的详细信息，是一个列表
    for item in dict_ticket.get(key):  # 根据可以获取值，dict_ticket[key]
        print(item, end='    ')
    print()  # 换行

#  输入用户的购票车次
train_no = input('请输入要购买的车次：')  # 需要考虑车次不存在的情况
#  根据key获取值
info = dict_ticket.get(train_no, '车次不存在')  # 妙啊
if info != '车次不存在':
    person = input('请输入乘车人，如果是多位乘车人使用逗号分开:')
    # 获取车i的出发站-到达站，还有出发时间。
    s = info[0] + ' ' + info[1] + '开'
    print('您已购买了车次' + train_no + '请' + person + '尽快取票上车')
else:
    print('车次不存在')

