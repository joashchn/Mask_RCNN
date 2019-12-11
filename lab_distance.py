import math
import cv2
import numpy as np
import os


def distance(x1, x2):
    x1, y1, z1 = x1
    x2, y2, z2 = x2
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    dis = math.sqrt(dx * dx + dy * dy + dz * dz)
    return dis


if __name__ == '__main__':

    file_dir = '/Users/oyo01135/PycharmProjects/Mask_RCNN/images/test/'
    x_t = []
    y_t = []
    for f in os.listdir(file_dir):
        img = cv2.imread(file_dir + f)
        img_t = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        img_t1 = img_t[:, :, 0]
        img_t2 = img_t[:, :, 1]
        img_t3 = img_t[:, :, 2]

        t1 = np.mean(img_t1) / 255.
        t2 = np.mean(img_t2) / 255.
        t3 = np.mean(img_t3) / 255.

        x = [t1, t2, t3]
        x_t.append(x)
        y_t.append(f)

    # 例如第3张照片和第5张的lab差值
    print(y_t)
    dis = distance(x_t[0], x_t[1])
    print('dis %s' % (dis))
