import os
path = os.getcwd()
print('当前路径为：{}'.format(path))
befor_path = os.path.dirname(path)
print('上一层目录为{}'.format(befor_path))
lst = os.walk(befor_path)
print(lst, type(lst))
for root, dirname, files in lst:
    # print('1:', root)
    # print('2:', dirname)
    # print('3:', files)
    # # 递归的显示出所有的目录，和目录下的目录与文件。
    # # 没一层都放在一个列表当中。
    print('-------------------')
    for dir in dirname:
        print(os.path.join(root, dir))
        # 查看当前目录下有多少个子目录
    for file in files:
        print(os.path.join(root, file))
        # 查看所有文件
