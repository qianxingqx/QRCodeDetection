# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:22:04 2023

@author: du
"""
import os
import cv2
import tqdm

import numpy as np
from os.path import join as opj
from pyzbar.pyzbar import decode
from QRCode_detection import os_makedirs, cv_imshow, QRCode_detection

if __name__ == "__main__":
    root_path = r"E:\Dataset\QRCodeDataset\original_selected_image"
    edge_pixels = 0  # 保留的边缘像素点个数

    # 裁剪二维码保存的位置
    all_qrcode_edge_path = r"E:\Dataset\QRCodeDataset\qrcode_edge_image"
    os_makedirs(all_qrcode_edge_path)

    # 扫描文件夹
    folder_list = os.listdir(root_path)
    for folder_i in folder_list:
        print("\n--Label:", folder_i)
        img_list = os.listdir(opj(root_path, folder_i))

        # 遍历文件夹内所有图片
        for img_i in tqdm.tqdm(img_list):
            # 读取图片
            img_array = cv2.imread(opj(root_path, folder_i, img_i), -1)

            # 识别二维码位置
            img_rect, img_warped, img_qrcode = QRCode_detection(img_array, edge_pixels=edge_pixels)
            # cv_imshow(img_rect, "img_rect")
            # cv_imshow(img_warped, "img_warped")
            # cv_imshow(img_qrcode, "img_qrcode")
            # 如果不存在二维码
            if img_rect is None:
                print("Warning: there are no QRCode in the image:", opj(root_path, folder_i, img_i))
                continue
            else:
                # 存在二维码
                # 保存裁剪旋转调整方向后的二维码图片
                qrcode_folder_i = opj(all_qrcode_edge_path, folder_i)
                os_makedirs(qrcode_folder_i)
                if not os.path.exists(opj(qrcode_folder_i, img_i)):
                    cv2.imwrite(opj(qrcode_folder_i, img_i), img_qrcode)
