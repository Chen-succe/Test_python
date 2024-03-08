# 列出指定目录下的所有python文件

import os

path = os.getcwd()
print(path)
lst = os.listdir(path)
# print(lst)
for i in lst:
    if i.endswith('.py'):
        # print(i)
        path_file = os.path.join(path, i)
        print(path_file, os.path.exists(path_file))