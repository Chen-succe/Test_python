import shutil
from PIL import Image
import os
import os.path
import glob
import time
from shutil import copyfile
import random
from random import sample



# 输出正样本误拒0.2%下的阈值
# score = []
# with open('./huokai_repvggb0_ep52_acc9827_pos_res.txt', 'r') as f:
#     for line in f: # enumrate all lines in this text file sequantially
#         each_line = line.strip('\n')
#         score.append(each_line.split(',')[-2])
#         #score = each_line.split(',')[-2]
#         score.sort(key=float)
#     print(score[7983])


# 输出各类型负样本误拒0.2%下的检出率
d = {}
with open('./crazytalk_new.txt', 'r') as f:
    for line in f: # enumrate all lines in this text file sequantially
        each_line = line.strip('\n')
        score = each_line.split(',')[-2]
        key = each_line.split(',')[-1]

        for i in [score]:
            # print(i)
            d.setdefault(key,[]).append(i)
    #print(d)
    result_dict = {}
    for key, values in d.items():  
        count_right = len([value for value in values if float(value) > 0.9994304776191711])
        count_all = len(values)
        result = format((count_right / count_all),'.2%')
        result_dict[key] = result
    print(result_dict)
            
