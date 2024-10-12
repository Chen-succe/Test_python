import cv2
import os
from tqdm import tqdm
from PIL import Image
# items = []
train_txt = '/mnt/mfs2/ailun.li/data_txt/all/huanlian/newlabel/huanlian_val_newlabel.txt'
# train_txt = '/mnt/mfs2/zhao.xie/data_list/swapface_jinghui/train_w.txt'


NAME = set()
output_dir = '/data/data1/haoyu.wang/huanlian_resize_new/val_PIL_224'
os.makedirs(output_dir,exist_ok=True)
txt = []
failed = 0
again = 0
with open('data_txts/huanlian_val_PIL_resize224_new.txt', 'w+') as fw:
    with open('data_txts/8g4/huokai_test_PIL_resize224_positive.txt', 'r') as f:
        for l in tqdm(f.readlines()):
            txt.append(l)
            fw.write(l)
    with open(train_txt, 'r') as f:
        for k, l in enumerate(tqdm(f.readlines())):
            strs = l.strip().split('<blank>')
            if int(strs[1]) == 0:
                continue
            img_path = strs[0]
            name = str(k)+"_"+os.path.basename(strs[0]).split('.')[0]+'.png'
            save_dir = os.path.join(output_dir,strs[1])
            os.makedirs(save_dir,exist_ok=True)
            if not os.path.exists(os.path.join(save_dir,name)):
                try:
                    img= Image.open(img_path).convert('RGB')
                    img = img.resize((224,224),resample=Image.BILINEAR)
            #         # img= cv2.imread(img_path)
            #         # img = cv2.resize(img,(224,224))
            #         # cv2.imwrite(os.path.join(output_dir,name),img)
                    img.save(os.path.join(save_dir,name))
                    fw.write(os.path.join(save_dir,name)+"<blank>"+strs[1]+'\n')
                    txt.append(os.path.join(output_dir,name)+"<blank>"+strs[1])
                except:
                    print(f'Read error {name}')
                    failed+=1
            #         pass
            else:
                again+=1
                # fw.write(l)
                print("Again:",os.path.join(save_dir,name))

            # if os.path.exists(os.path.join(output_dir, name)):
                #     txt.append(name)
                #     fw.write(os.path.join(output_dir, name) + "<blank>" + strs[1] + '\n')
print(len(txt))
print(failed)
print(again)
# with open('data_txts/all_train_PIL_resize320.txt','a+') as f:
#     for t in txt:
#         f.write(t+'\n')


# 在PIL resize基础上重新做数据集

# train_txt = 'data_txts/huokai_train_PIL_resize224.txt'
# txt = []
# with open(train_txt, 'r') as f:
#     for k, l in enumerate(tqdm(f.readlines())):
#         strs = l.strip().split('<blank>')
#         img_path = strs[0]
#         label = strs[1]
#         if label!= '0':
#             continue
#         txt.append(img_path+'<blank>'+label)
#
# with open('data_txts/huokai_train_PIL_resize224_positive.txt','w+') as f:
#     for t in tqdm(txt):
#         f.write(t+'\n')
#
#


# all_txt ='/mnt/mfs2/ailun.li/data_txt/all/all_val.txt'
# huokai_txt = '/mnt/mfs2/ailun.li/data_txt/huokai/huokai_test.txt'
# huokai_txt ='data_txts/huanlian_val_PIL_resize224.txt'
# huokai_txt = '/mnt/mfs2/ailun.li/data_txt/all/huanlian/newlabel/huanlian_test_newlabel.txt'
# huokai_txt = '/mnt/mfs2/ailun.li/data_txt/all/all_val.txt'
# output_dir = '/data/data1/haoyu.wang/huanlian_resize/val_PIL'
# os.makedirs(output_dir,exist_ok=True)
#
# huokai_list = []
# huokai_labels = []
# labels = set()
# huokai_hash = dict()
# all_in = 0
# with open(huokai_txt, 'r') as f:
#     for k, l in enumerate(tqdm(f.readlines())):
#         strs = l.strip().split('<blank>')
#         huokai_list.append(strs[0])
#         huokai_labels.append(int(strs[1]))
#         labels.add(int(strs[1]))
#         huokai_hash[strs[0]] = int(strs[1])
#         if 'Talkr' in strs[0] or 'talkr' in strs[0]:
#             all_in+=1
#             print(strs)
#         # if int(strs[1])<23:
#         #     print(strs)
# print(all_in)
#
#
# all_list = []
# all_labels= []
# all_hash = dict()
# huanlian_list = []
# huanlian_labels=  []
# same_labels= []
# fail = 0
# in_all = 0
# with open(all_txt, 'r') as f:
#     for k, l in enumerate(tqdm(f.readlines())):
#         strs = l.strip().split('<blank>')
#         img_path = strs[0]
#         all_list.append(img_path)
#         # all_hash[img_path] = int(strs[1])
#         if huokai_hash.get(img_path,-1)!=-1:
#             in_all+=1
#             continue
# print(in_all)
        # else:

        # if fail==0:
        #     print(k)
        #     fail=1
        # name = str(k)+'_'+os.path.basename(strs[0]).split('.')[0]+'.png'
        # if k<len(huokai_labels):
        #     if os.path.exists(os.path.join('/data/data1/haoyu.wang/huohua_resize/train_PIL', name)):
        #         all_list.append(os.path.join('/data/data1/haoyu.wang/huohua_resize/train_PIL', name) + "<blank>" + strs[1])
        #     else:
        #         print(os.path.join('/data/data1/haoyu.wang/huohua_resize/train_PIL', name))
        # else:
        # try:
        #     img = Image.open(img_path).convert('RGB')
        #     img = img.resize((224, 224), resample=Image.BILINEAR)
        #     # assert not os.path.exists(os.path.join(output_dir, name))
        #     img.save(os.path.join(output_dir, name))
        #     # all_list.append(os.path.join(output_dir, name) + "<blank>" + strs[1])
        #     huanlian_list.append(os.path.join(output_dir, name) + "<blank>" + strs[1])
        # except:
        #     fail+=1
        #     pass
#
# print(f"fail {fail}")
# # with open('data_txts/all_train_PIL_resize224.txt','w+') as f:
# #     for t in all_list:
# #         f.write(t+'\n')
# with open('data_txts/huanlian_val_PIL_resize224.txt','w+') as f:
#     for t in huanlian_list:
#         f.write(t+'\n')


from collections import Counter
# for mode  in ['test','val']:
#     # ori_txt = f'/mnt/mfs2/ailun.li/data_txt/all/all_{mode}.txt'
#     # label = []
#     # with open(ori_txt, 'r') as f:
#     #     for k, l in enumerate(tqdm(f.readlines())):
#     #         strs = l.strip().split('<blank>')
#     #         img_path = strs[0]
#     #         label.append(int(strs[1]))
# huanlian_list = []
# huokai_txt = 'data_txts/huokai_test_PIL_resize224.txt'
# with open(huokai_txt, 'r') as f:
#     for k, l in enumerate(tqdm(f.readlines())):
#         strs = l.strip().split('<blank>')
#         img_path = strs[0]
#         huanlian_list.append( img_path+ "<blank>" + strs[1])
#     resize_txt = f'data_txts/huanlian_{mode}_PIL_resize224.txt' # data_txts/huanlian_val_PIL_resize224.txt
# with open(resize_txt, 'r') as f:
#     for k, l in enumerate(tqdm(f.readlines())):
#         strs = l.strip().split('<blank>')
#         img_path = strs[0]
#         huanlian_list.append( img_path+ "<blank>" + strs[1])
#     with open(f'data_txts/all_{mode}_PIL_resize224.txt','w+') as f:
#         for t in huanlian_list:
#             f.write(t+'\n')
#     print(f'{mode}, {len(huanlian_list)}')
    # print(f'{mode}')
    # l_c = Counter(label)
    # ls_c = Counter(labels)
    # keys = list(ls_c.keys())
    # keys.sort()
    #
    # for k in keys:
    #     print(k,l_c[k],ls_c[k])
    # print('     *****************************     ')
    #

# print('huokai: ')
# print(Counter(huokai_labels))
#
# print('all: ')
# print(Counter(all_labels))
#
# print('Huanlian:')
# print(Counter(huanlian_labels))
#
# print('Save:')
# print(Counter(same_labels))