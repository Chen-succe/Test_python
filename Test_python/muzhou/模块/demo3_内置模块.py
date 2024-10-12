# coding:utf-8

import sys
import time
import urllib.request
import schedule

print(sys.getsizeof(24))  # 获取对象所占内存大小
print(sys.getsizeof(40))  # 28
print(sys.getsizeof(True))  # 28
print(sys.getsizeof(False))  # 24
print(time.time())
print(time.localtime(time.time()))


# print(urllib.request.urlopen('http://www.baidu.com').read())

def job():
    print('哈哈哈')

schedule.every(3).seconds.do(job)
while True:
    schedule.run_pending()
    time.sleep(1)