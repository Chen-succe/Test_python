
# 二进制方式练习，b要结合r和w使用。
file = open('xjpic.jpg', 'rb')
target_file = open('copyxjpic.jpg', 'wb')
target_file.write(file.read())
file.close()
target_file.close()