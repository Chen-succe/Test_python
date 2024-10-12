import os

import albumentations as A
from tqdm import tqdm
import cv2
import os
import random
from collections import Counter

# train_txt ='/mnt/mfs2/ailun.li/data_txt/all/huanlian/newlabel/huanlian_train_newlabel.txt'
# txt = []
# labels = []
# # output = './visualization/data_augmentation/'
# # os.makedirs(output,exist_ok=True)
# with open(train_txt, 'r') as f:
#     for k, l in enumerate(tqdm(f.readlines())):
#         strs = l.strip().split('<blank>')
#         txt.append([strs[0],int(strs[1])])
#         labels.append(int(strs[1]))
# print(Counter(labels))
# random.shuffle(txt)
# for img_path,label in txt:
#     if label != 0:
#         continue
#     print(img_path)
#     img = cv2.imread(img_path)
#     name = os.path.basename(img_path)
#     for quality in range(10,100,10):
#         jpeg = A.JpegCompression(quality_lower=quality,quality_upper=quality,always_apply=True,p=1)
#         img_jpeg = jpeg(image = img)['image']
#         cv2.imwrite(os.path.join(output,name[:-4]+f'_jpeg{quality}.png'),img_jpeg)
#         cv2.imwrite(os.path.join(output, name[:-4] + f'_jpeg{quality}_diff.png'), img-img_jpeg)
#     break
#
# for img_path,label in txt:
#     if label == 0:
#         continue
#     print(img_path)
#     img = cv2.imread(img_path)
#     name = os.path.basename(img_path)
#     for quality in range(10,100,10):
#         jpeg = A.JpegCompression(quality_lower=quality,quality_upper=quality,always_apply=True,p=1)
#         img_jpeg = jpeg(image = img)['image']
#         cv2.imwrite(os.path.join(output,name[:-4]+f'_jpeg{quality}.png'),img_jpeg)
#         cv2.imwrite(os.path.join(output, name[:-4] + f'_jpeg{quality}_diff.png'), img-img_jpeg)
#     break
#
# def cal(th,s):
#     x = (np.log((1 - th) / th)) / -2 / s
#     print(x)
#     xita = np.arccos(x) * 2 / np.pi * 90
#     print(xita)
# if __name__ == '__main__':
#
#     import numpy as np
#     th = 0.8370
#     s = 2


"""
from huokai_train_PIL_resize224_8g8 filter positive,crazytalk,muglife
"""

# train_txt = 'data_txts/huokai_train_PIL_resize224.txt'
# test_txt = 'data_txts/huokai_test_PIL_resize224.txt'
#
#
#
# train = []
# with open('data_txts/8g4/huokai_train_PIL_resize224_positive.txt','w+') as p:
#     with open('data_txts/8g4/huokai_train_PIL_resize224_crazytalk.txt','w+') as c:
#         with open('data_txts/8g4/huokai_train_PIL_resize224_muglife.txt', 'w+') as m:
#             with open(train_txt, 'r') as f:
#                 for l in tqdm(f.readlines()):
#                     strs = l.strip().split('<blank>')
#                     label = int(strs[1])
#                     if label==0:
#                         p.write(l)
#                     elif label==12:
#                         c.write(l)
#                     elif label == 22:
#                         m.write(l)
#
#
# test = []
# with open('data_txts/8g4/huokai_test_PIL_resize224_positive.txt', 'w+') as p:
#     with open('data_txts/8g4/huokai_test_PIL_resize224_crazytalk.txt', 'w+') as c:
#         with open('data_txts/8g4/huokai_test_PIL_resize224_muglife.txt', 'w+') as m:
#             with open(test_txt, 'r') as f:
#                 for l in tqdm(f.readlines()):
#                     strs = l.strip().split('<blank>')
#                     label = int(strs[1])
#                     if label == 0:
#                         p.write(l)
#                     elif label == 12:
#                         c.write(l)
#                     elif label == 22:
#                         m.write(l)
#
# # with open(test_txt, 'r') as f:
# #     for l in tqdm(f.readlines()):
# #         strs = l.strip().split('<blank>')
# #         if int(strs[1])==12 or int(strs[1])==22:
# # #             test.append(l)
# print(len(train))
# print(len(test))



"""
from crazytalk filter new&old
"""
# train_txt = 'data_txts/8g8/huokai_test_PIL_resize224_crazytalk.txt'

# train_old = []
# train_new = []
# with open('data_txts/8g4/huokai_test_PIL_resize224_renlian11000.txt','w+') as co:
#     with open('data_txts/8g8/huokai_test_PIL_resize224_renlian11000.txt') as f:
#         for l  in tqdm(f.readlines()):
#             co.write(l.replace('/data/haoyu.wang','/data/data1/haoyu.wang'))
#             train_old.append(l)
# with open('data_txts/8g4/huokai_train_PIL_resize224_renlian11000.txt', 'w+') as cn:
#     with open('data_txts/8g8/huokai_train_PIL_resize224_renlian11000.txt') as f:
#         for l  in tqdm(f.readlines()):
#             cn.write(l.replace('/data/haoyu.wang','/data/data1/haoyu.wang'))
#             train_new.append(l)
#         with open(train_txt, 'r') as f:
#             for l in tqdm(f.readlines()):
#                 if 'test_new' in l:
#                     cn.write(l)
#                     train_new.append(l)
#                 elif 'test_old' in l:
#                     co.write(l)
#                     train_old.append(l)
#
# print(len(train_new))
# print(len(train_old))

# """
# from crazytalk new filter clean
# """
# train_crazytalk_new_all_txt = 'data_txts/8g4/huokai_train_PIL_resize224_crazytalknew.txt'
# train_crazytalk_new_all = dict()
# with open(train_crazytalk_new_all_txt, 'r') as f:
#     for l in tqdm(f.readlines()):
#         strs = l.strip().split('<blank>')
#         base_name = os.path.basename(strs[0]).split('.')[0]
#         name = '_'.join(base_name.split('_')[1:])
#         train_crazytalk_new_all[name] = l
#
# #
# train_crazytalk_new_clean_txt = '/mnt/mfs2/ailun.li/data_txt/clean/crazytalk_only_model/crazyclean_train.txt'
# train_crazytalk_new_clean = []
# with open(train_crazytalk_new_clean_txt, 'r') as f:
#     with open('data_txts/8g4/huokai_train_PIL_resize224_crazytalknew_clean.txt','w+') as c:
#         for l in tqdm(f.readlines()):
#             strs = l.strip().split('<blank>')
#             base_name = os.path.basename(strs[0]).split('.')[0]
#             cur_path = train_crazytalk_new_all.get(base_name, None)
#             if cur_path:
#                 train_crazytalk_new_clean.append(cur_path)
#                 c.write(cur_path)
# print('Crazytalk new train clean:' , len(train_crazytalk_new_clean))
#
#
# test_crazytalk_new_all_txt = 'data_txts/8g4/huokai_test_PIL_resize224_crazytalknew.txt'
# test_crazytalk_new_all = dict()
# with open(test_crazytalk_new_all_txt, 'r') as f:
#     for l in tqdm(f.readlines()):
#         strs = l.strip().split('<blank>')
#         base_name = os.path.basename(strs[0]).split('.')[0]
#         name = '_'.join(base_name.split('_')[1:])
#         test_crazytalk_new_all[name] = l
#
#
# test_crazytalk_new_clean_txt = '/mnt/mfs2/ailun.li/data_txt/clean/crazytalk_only_model/crazyclean_test.txt'
# test_crazytalk_new_clean = []
# with open(test_crazytalk_new_clean_txt, 'r') as f:
#     with open('data_txts/8g4/huokai_test_PIL_resize224_crazytalknew_clean.txt','w+') as c:
#         for l in tqdm(f.readlines()):
#             strs = l.strip().split('<blank>')
#             base_name = os.path.basename(strs[0]).split('.')[0]
#             cur_path = test_crazytalk_new_all.get(base_name, None)
#             if cur_path:
#                 test_crazytalk_new_clean.append(cur_path)
#                 c.write(cur_path)
# print('Crazytalk new test clean:' , len(test_crazytalk_new_clean))
# #


"""
from positive filter renlian11000
"""
# train_positive_new_all_txt = 'data_txts/8g8/huokai_train_PIL_resize224_positive.txt'
# train_positive_new_all = dict()
# with open(train_positive_new_all_txt, 'r') as f:
#     for l in tqdm(f.readlines()):
#         strs = l.strip().split('<blank>')
#         base_name = os.path.basename(strs[0]).split('.')[0]
#         name = '_'.join(base_name.split('_')[1:])
#         train_positive_new_all[name] = l
#
#
# train_renlian11000_old_txt = '/mnt/mfs2/ailun.li/data_txt/crazytalk/crazytalk_old/crazytalk_old+renlian11000_train.txt'
# train_renlian11000 = []
# with open(train_renlian11000_old_txt, 'r') as f:
#     with open('data_txts/8g8/huokai_train_PIL_resize224_renlian11000.txt','w+') as c:
#         for l in tqdm(f.readlines()):
#             strs = l.strip().split('<blank>')
#             if int(strs[1])!=0:
#                 continue
#             base_name = os.path.basename(strs[0]).split('.')[0]
#             cur_path = train_positive_new_all.get(base_name, None)
#             if cur_path:
#                 train_renlian11000.append(cur_path)
#                 c.write(cur_path)
# print('renlian11000 train:' , len(train_renlian11000))
#
#
# test_positive_new_all_txt = 'data_txts/8g8/huokai_test_PIL_resize224_positive.txt'
# test_positive_new_all = dict()
# with open(test_positive_new_all_txt, 'r') as f:
#     for l in tqdm(f.readlines()):
#         strs = l.strip().split('<blank>')
#         base_name = os.path.basename(strs[0]).split('.')[0]
#         name = '_'.join(base_name.split('_')[1:])
#         test_positive_new_all[name] = l
#
#
# test_renlian11000_old_txt = '/mnt/mfs2/ailun.li/data_txt/crazytalk/crazytalk_old/crazytalk_old+renlian11000_test.txt'
# test_renlian11000 = []
# with open(test_renlian11000_old_txt, 'r') as f:
#     with open('data_txts/8g8/huokai_test_PIL_resize224_renlian11000.txt','w+') as c:
#         for l in tqdm(f.readlines()):
#             strs = l.strip().split('<blank>')
#             if int(strs[1])!=0:
#                 continue
#             base_name = os.path.basename(strs[0]).split('.')[0]
#             cur_path = test_positive_new_all.get(base_name, None)
#             if cur_path:
#                 test_renlian11000.append(cur_path)
#                 c.write(cur_path)
#
# print('renlian11000 test:' , len(test_renlian11000))


# from glob import glob
# '''
# copy w_train from mnt 8g8
# '''
# w_train = '/data/haoyu.wang/qita_pos/w_test/'
# w_trains = glob(w_train+'*.png')
# w_trains.sort()
# with open('data_txts/8g8/Wtrain_PIL_resize224.txt','w+') as f:
#     for w in tqdm(w_trains):
#         f.write(w+'<blank>'+'0\n')

"""
maks huanlian train,test,val
"""
train_huanlian_txt = 'data_txts/all_train_PIL_resize224.txt'
huokai_train_txt =  'data_txts/huokai_train_PIL_resize224.txt'
huokai_test_txt =  'data_txts/huokai_test_PIL_resize224.txt'

test_huanlian_txt = 'data_txts/all_test_PIL_resize224.txt'
val_huanlian_txt = 'data_txts/all_val_PIL_resize224.txt'
# positive = []

huokai_train = dict()
huokai_test = dict()
with open(huokai_train_txt) as f:
    for l in tqdm(f.readlines()):
        strs = l.strip().split('<blank>')
        if strs[1] not in huokai_train:
            huokai_train[strs[1]] = []
        huokai_train[strs[1]].append(strs[0])
with open(huokai_test_txt) as f:
    for l in tqdm(f.readlines()):
        strs = l.strip().split('<blank>')
        if strs[1] not in huokai_test:
            huokai_test[strs[1]] = []
        huokai_test[strs[1]].append(strs[0])

train_huanlian = []
test_huanlian = []
val_huanlian = []
map_label = {0:0,16:1,19:5,20:4,21:2,24:3,25:6,26:7,27:8,28:9,29:10,30:11,
             31:12,32:13,33:14,34:15,35:16,36:17,37:18,38:19,39:20}
# pos_huanlian = []
with open(train_huanlian_txt) as f:
    for l in tqdm(f.readlines()):
        strs = l.strip().split('<blank>')
        if int(strs[1]) in map_label.keys():
            if strs[1] =='0' or  huokai_train.get(strs[1],None) is None or strs[0] not in huokai_train[strs[1]]:
                train_huanlian.append(f"{strs[0]}<blank>{map_label[int(strs[1])]}\n")
# with open(val_huanlian_txt) as f:
#     for l in tqdm(f.readlines()):
#         strs = l.strip().split('<blank>')
#         if int(strs[1]) in map_label.keys():
#             if strs[1] =='0' or huokai_test.get(strs[1], None) is None or strs[0] not in huokai_test[strs[1]]:
#                 val_huanlian.append(f"{strs[0]}<blank>{map_label[int(strs[1])]}\n")
# with open(test_huanlian_txt) as f:
#     for l in tqdm(f.readlines()):
#         strs = l.strip().split('<blank>')
#         if int(strs[1]) in map_label.keys() or int(strs[1])>100:
#             if  strs[1] =='0' or  huokai_test.get(strs[1], None) is None or strs[0] not in huokai_test[strs[1]]:
#                 test_huanlian.append(f"{strs[0]}<blank>{map_label.get(int(strs[1]),int(strs[1]))}\n")

print(len(train_huanlian))
print(len(test_huanlian))
print(len(val_huanlian))
with open('data_txts/huanlian_train_PIL_resize224_new.txt.','w+') as w:
    for p in train_huanlian:
        w.write(p)
# with open('data_txts/huanlian_test_PIL_resize224_.txt.','w+') as w:
#     for p in test_huanlian:
#         w.write(p)
# with open('data_txts/huanlian_val_PIL_resize224_.txt.','w+') as w:
#     for p in val_huanlian:
#         w.write(p)




