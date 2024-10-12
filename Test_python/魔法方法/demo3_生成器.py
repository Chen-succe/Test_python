def gen(num):
    while num > 0:
        temp = yield num
        print('我是中间值:' + str(temp))
        num -= 1
        # return 'hi'
    print('end')
    try:
        return '从exception中得到的'
    except Exception as e:
        print(e)


g = gen(5)
# for i in g:
#     print(i)

# a = next(g)
# print(next(g))
# print(next(g))
# print(next(g))
# print(next(g))
# print(next(g))

# import dis
# print(dis.dis(g))

first = next(g)
print(first)
print(g.send(10))