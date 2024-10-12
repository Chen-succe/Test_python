# coding:utf-8

# 创建一个列表，用于存储入库的商品信息
list_shop = []
for i in range(2):
    goods= input('请输入商品编号进行商品入库，每次只能输入一讲商品，输入q退出:')
    list_shop.append(goods)

# 输出所有的商品信息
for item in list_shop:
    print(item)

# 创建一个空列表，存储添加到购物车的商品
list_shopping_car = []

# 上面range我们知道共有多少商品，这里添加购物车时，我们不知道要购买多少，使用while循环，输入q停止。
while True:
    num = input('请输入要购买的商品编号:')
    flag = False  # 代表没有商品的情况
    # 遍历列表，查询要购买的商品是否存在。编号是前四位
    for item in list_shop:
        if num == item[:4]:
            list_shopping_car.append(item)  # 如果存在，添加到购物车列表中
            print('商品已添加到购物车，请继续添加，输入q退出')
            flag = True  # 代表商品已找到
            break  # 退出的是for循环

    if num == 'q':
        break  # 退出的是while循环

    if not flag:  # 在退出前判断是否找到该商品
        print('该商品不存在')

print('您的购物车中已选择的商品为:')
list_shopping_car.reverse()
for item in list_shopping_car:
    print(item)
