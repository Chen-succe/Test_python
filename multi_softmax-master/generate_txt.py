import os
import cv2
import random

KEYS = {
    'w_test': 1,
    'zhaji_languang': 2,
    'zhaji_zhengchang': 3,
    'ZY_wuju_baidu': 4,
    'ZY_wuju_KS': 5,
}


def generate_datalist(txt_dir):
    input_path = '/media/realai/9_2T/positive_new/test_beiyong/positive_qita/crop'
    labels = []
    items = []
    for d, n, filename in os.walk(input_path):
        for f in filename:
            if not f.startswith('.') and not f.endswith(".mp4"):
                img_pth = (os.path.join(d, f))
                # print(img_pth)
                for key in KEYS.keys():
                    if key in img_pth:
                        multi_label = KEYS[key]
                        labels.append(multi_label)
                        items.append(img_pth + '<blank>' + str(multi_label) + '\n')
        annos_pth = os.path.join(txt_dir, 'positive_qita.txt')
        with open(annos_pth, 'w') as f:
            f.writelines(items)


if __name__ == '__main__':
    txt_dir = '/home/realai/ailun/norm_cls'
    generate_datalist(txt_dir)
