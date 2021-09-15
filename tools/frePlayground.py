import cv2
import numpy as np
import os
from skimage.exposure import match_histograms
import glob

if __name__ == '__main__':
    IMG_D0_PATH = "../data/cityscape/VOC2007/JPEGImages"
    IMG_D1_PATH = "../DataSet/SIM10K/VOC2012/JPEGImages"
    D1_PATH = "../DataSet/SIM10K/VOC2012/JPEGImages/3386352.jpg"
    MASK_PATH = "./SaveFile/nparray/img_D0_mask.npy"
    SAVE_PATH = "./SaveFile/image/colortry"
    R_DIV = 64
    TRY_DIV = 3
    D1_SET_FLAG = 0

    print("frePlayground")

    img_D0_Lists = glob.glob(IMG_D0_PATH + '/*.jpg')
    img_D0_basenames = [] # e.g. 100.jpg
    for item in img_D0_Lists:
        img_D0_basenames.append(os.path.basename(item))
    img_D0_names = [] # e.g. 100
    for item in img_D0_basenames:
        temp1, temp2 = os.path.splitext(item)
        img_D0_names.append(temp1)
    
    if D1_SET_FLAG == 1:
        img_D1_Lists = glob.glob(IMG_D1_PATH + '/*.jpg')
        img_D1_basenames = [] # e.g. 100.jpg
        for item in img_D1_Lists:
            img_D1_basenames.append(os.path.basename(item))
        img_D1_names = [] # e.g. 100
        for item in img_D1_basenames:
            temp1, temp2 = os.path.splitext(item)
            img_D1_names.append(temp1)

    if  not os.path.exists(SAVE_PATH):#如果路径不存在
        os.makedirs(SAVE_PATH)
    
    for img_idx,img in enumerate(img_D0_names):
        print("Process img: %s/%s" % (img_idx,len(img_D0_names)))
        img_dir = IMG_D0_PATH + '/' + img + ".jpg"
        img_D0 = cv2.imread(img_dir)
        img_h, img_w, img_c = img_D0.shape
        r_max = (img_w**2 + img_h**2)**(1/2)
        r_l = np.linspace(0,r_max,r_max//R_DIV)

        img_mask = np.zeros((R_DIV,img_h,img_w))
        img_mask_try = np.zeros((TRY_DIV,img_h,img_w))
        img_end = np.zeros((R_DIV,img_h,img_w,img_c))
        img_D0_dct = np.zeros((img_h,img_w,img_c))

        # mask 的生成
        if not os.path.exists(MASK_PATH):
            print("create mask file..")
            for t in range(R_DIV):
                print("r: %s" %t)
                for i in range(img_h):
                    for j in range(img_w):
                        if (r_max//R_DIV)*t <= (i**2+j**2)**0.5 <= (r_max//R_DIV)*(t+1):
                            img_mask[t][i][j] = 1
                        else:
                            img_mask[t][i][j] = 0
            np.save("./SaveFile/nparray/img_D0_mask.npy",img_mask)
                # cv2.imwrite("img_mask.jpg",img_mask*255)
        else:
            print("Read mask file..")
            img_mask = np.load(MASK_PATH)
            print("read %s finished!" % MASK_PATH)
        
        img_mask_try[0] = img_mask[0:2].sum(axis=0)
        img_mask_try[1] = img_mask[2:32].sum(axis=0)
        img_mask_try[2] = img_mask[32:64].sum(axis=0)

        # D0 DCT
        for c in range(img_c):
            img_D0_dct[:,:,c] = cv2.dct(np.float32(img_D0[:, :, c]))
        # D0 IDCT
        for t in range(TRY_DIV):
            print("D1->r: %s" %t)
            for c in range(img_c):
                img_dct_m = img_D0_dct[:,:,c]*img_mask_try[t]
                
                img_idct = cv2.idct(img_dct_m)
                img_end[t,:,:,c] = img_idct
                # img_idct = np.uint8(img_idct/img_idct.max()*255)
                # img_show[:, :, c] = cv2.equalizeHist(img_idct)
            # cv2.imwrite("SaveFile/image/dct_%s.jpg" %t,img_end[t])
            # cv2.imwrite("img.jpg",img)
        
        # Domain1
        if D1_SET_FLAG == 1:
            img_D1_dir = IMG_D1_PATH + '/' + img_D1_names[img_idx%(len(img_D1_names))] + ".jpg"
            img_D1 = cv2.imread(img_D1_dir)
        else:
            img_D1 = cv2.imread(D1_PATH)
        img_D1 = cv2.resize(img_D1, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        img_D1_end = np.zeros((R_DIV,img_h,img_w,img_c))
        img_D1_dct = np.zeros((img_h,img_w,img_c))

        for c in range(img_c):
            img_D1_dct[:,:,c] = cv2.dct(np.float32(img_D1[:, :, c]))
        cv2.imwrite("SaveFile/image/img_D1_dct.jpg", img_D1_dct)
        
        for t in range(TRY_DIV):
            print("D1->r: %s" %t)
            for c in range(img_c):
                img_D1_dct_m = img_D1_dct[:,:,c]*img_mask_try[t]
            
                img_idct = cv2.idct(img_D1_dct_m)
                img_end[t,:,:,c] = img_idct
                # img_idct = np.uint8(img_idct/img_idct.max()*255)
                # img_show[:, :, c] = cv2.equalizeHist(img_idct)
            # cv2.imwrite("SaveFile/image/Domain_1_%s.jpg" %t,img_end[t])

        # img_D01_dct = np.zeros((img_h,img_w,img_c))
        img_show = np.zeros((img_h,img_w,img_c))
        # for c in range(img_c):
        # img_D01_dct = match_histograms(img_D0_dct,img_D1_dct,multichannel=True)
        # img_D01_dct = style_transfer(img_D0_dct, img_D1_dct)
        # cv2.imwrite("SaveFile/image/idct.jpg", img_D01_dct)   # 保存融合后的特征图

        # 保存域变化后的全图
        # for c in range(img_c):
        #     img_show[:,:,c] = cv2.idct(img_D01_dct[:,:,c])
        # cv2.imwrite("SaveFile/image/D01.jpg", img_show)

        img_D0_dct_s = np.zeros((img_h,img_w,img_c))
        img_D1_dct_s = np.zeros((img_h,img_w,img_c))
        for c in range(img_c):
            img_D0_dct_s[:,:,c] = img_D0_dct[:,:,c]*img_mask_try[0]
            img_D1_dct_s[:,:,c] = img_D1_dct[:,:,c]*img_mask_try[0]

        cv2.imwrite("SaveFile/image/colortry/D0_dct.jpg", img_D0_dct_s)
        cv2.imwrite("SaveFile/image/colortry/D1_dct.jpg", img_D1_dct_s)
        img_D01_dct_s = match_histograms(img_D0_dct_s,img_D1_dct_s,multichannel=True)
        cv2.imwrite("SaveFile/image/colortry/match_dct.jpg", img_D01_dct_s)

        # for c in range(img_c):
        #     img_D01_dct_s[:,:,c] = cv2.idct(img_D0_dct_s[:,:,c])
        # cv2.imwrite("SaveFile/image/colortry/match_l_idct.jpg", img_D01_dct_s)

        for c in range(img_c):
            img_D01_dct_s[:,:,c] = img_D0_dct[:,:,c]*img_mask_try[1] + img_D0_dct[:,:,c]*img_mask_try[2] + img_D01_dct_s[:,:,c]
        cv2.imwrite("SaveFile/image/colortry/add_dct.jpg", img_D01_dct_s)
        for c in range(img_c):
            img_show[:,:,c] = cv2.idct(img_D01_dct_s[:,:,c])
        cv2.imwrite(SAVE_PATH + "/%s.jpg" %img, img_show) 
