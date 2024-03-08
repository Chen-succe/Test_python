# os.path模块
import os.path
print(os.path.abspath('demo7_os点path模块.py'))
print(os.path.exists('demo7_os点path模块.py'))
print(os.path.join('D:\\D', 'demo7_os点path模块.py'))
print(os.path.split('D:/D/PythonFile/2024_test/Test_python/muzhou/最后part/demo7_os点path模块.py'))
# 将文件和目录拆分
print(os.path.splitext('D:/D/PythonFile/2024_test/Test_python/muzhou/最后part/demo7_os点path模块.py'))
# 将文件和后缀名进行拆分
print(os.path.basename('D:/D/PythonFile/2024_test/Test_python'))  # 提取出文件名
print(os.path.dirname('D:/D/PythonFile/2024_test/Test_python'))  # 提取出文件路径，不包括文件名
print(os.path.isdir('demo7_os点path模块.py'))
print(os.path.isdir('D:/D/PythonFile/2024_test/Test_python'))