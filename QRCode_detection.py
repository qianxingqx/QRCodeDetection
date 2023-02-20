# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:18:13 2023
Description: 对输入的图片进行预处理，提取出图片中的二维码，并裁剪得到二维码区域，然后旋转至正确方向
Notes:
    1. 由于采集的数据集有 HEIC 和 PNG 格式，所以该程序还用于将所有图片统一至 png 格式并保存
"""
import os
import cv2
import tqdm

import numpy as np
from os.path import join as opj
from pyzbar.pyzbar import decode


def cv_imshow(img_array, img_name="Image"):
    """ CV 绘图显示
    """
    cv2.namedWindow(img_name, 0)  # 代表窗口可伸缩，否则 imshow 可能会显示不全
    cv2.imshow(img_name, img_array)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def os_makedirs(path):
    """ 不存在则创建文件夹
    """
    if not os.path.exists(path):
        os.makedirs(path)


def QRCode_detection(img_array, edge_pixels=5):
    """ 检测图片中的二维码，只允许检测单个
    Args:
        img_array: nd.array
        edge_pixels: 保留的边缘像素点个数
    Returns:
        img_rect: 在原图基础上用线段标记了二维码位置的图片
        img_warped: 裁剪出的原始二维码图片
        img_rotate: 将二维码旋转至统一的方向的图片
    """
    # 转换为灰度图像
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # 找到图像中的所有二维码
    # 可改为：test=pyzbar.decode(frame, ymbols=[pyzbar.ZBarSymbol.QRCODE])
    decoded_objs = decode(gray)

    # 如果至少找到了一个二维码，就对其进行处理
    if len(decoded_objs) > 0:
        if len(decoded_objs) > 1:
            print("There are {} QRCode in the image!".format(len(decoded_objs)))
        # 获取第一个二维码对象
        obj = decoded_objs[0]

        # 获取二维码的位置和大小
        rect = obj.rect

        # 获取二维码的四个角点
        points = obj.polygon
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))

        # 提取二维码的区域
        width = int(rect[2] * 1.5)
        height = int(rect[3] * 1.5)

        src_pts = np.float32([points[0][0], points[1][0], points[2][0], points[3][0]])
        # dst_pts = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        dst_pts = np.float32([[width - 1, height - 1], [width - 1, 0], [0, 0], [0, height - 1]])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # 将裁剪中心向左上角偏移，然后扩大 width 和 height 大小
        M[0, 2] += edge_pixels
        M[1, 2] += edge_pixels
        img_warped = cv2.warpPerspective(img_array, M, (width+edge_pixels*2, height+edge_pixels*2))

        # 将二维码位置标记在原图上
        img_rect = img_array.copy()
        line_color = (0, 0, 255)
        line_thickness = 3
        cv2.line(img_rect, (int(src_pts[0][0]), int(src_pts[0][1])), (int(src_pts[1][0]), int(src_pts[1][1])), color=line_color, thickness=line_thickness)
        cv2.line(img_rect, (int(src_pts[1][0]), int(src_pts[1][1])), (int(src_pts[2][0]), int(src_pts[2][1])), color=line_color, thickness=line_thickness)
        cv2.line(img_rect, (int(src_pts[2][0]), int(src_pts[2][1])), (int(src_pts[3][0]), int(src_pts[3][1])), color=line_color, thickness=line_thickness)
        cv2.line(img_rect, (int(src_pts[3][0]), int(src_pts[3][1])), (int(src_pts[0][0]), int(src_pts[0][1])), color=line_color, thickness=line_thickness)

        # 将二维码统一旋转至 DOWN 方向
        # LEFT |-, DOWN |_, RIGHT _|, UP -|
        img_warped_gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
        img_warped_decoded_objs = decode(img_warped_gray)
        if len(img_warped_decoded_objs) == 0:
            # print("No QR code found in the image.")
            return None, None, None
        img_warped_obj = img_warped_decoded_objs[0]
        # print(img_warped_obj.orientation)
        if img_warped_obj.orientation == "UP":
            img_rotate = np.rot90(img_warped, k=-3)  # 顺时针旋转 270
        elif img_warped_obj.orientation == "RIGHT":
            img_rotate = np.rot90(img_warped, k=-2)  # 顺时针旋转 180
        elif img_warped_obj.orientation == "DOWN":
            img_rotate = np.rot90(img_warped, k=-1)  # 顺时针旋转 90
        else:
            img_rotate = img_warped.copy()
        # cv_imshow(img_array)
        # cv_imshow(img_warped, 'w')
        # cv_imshow(img_rotate, 'r')

        # # 将二维码裁剪成正方形
        # if width > height:
        #     margin = int((width - height) * 0.2)  # 扩展20%的边缘
        #     img_cropped = img_rotate[:, (width-height)//2-margin:(width+height)//2+margin]
        # else:
        #     margin = int((height - width) * 0.2)  # 扩展20%的边缘
        #     img_cropped = img_rotate[(height-width)//2-margin:(height+width)//2+margin, :]
        # cv_imshow(img_rotate, "w")
        # cv_imshow(img_cropped, "c")

        return img_rect, img_warped, img_rotate
    else:
        # print("No QR code found in the image.")
        return None, None, None


if __name__ == "__main__":
    root_path = r"E:\Dataset\QRCodeDataset\all"

    all_png_path = r"E:\Dataset\QRCodeDataset\all_png"
    all_qrcode_path = r"E:\Dataset\QRCodeDataset\all_qrcode"
    os_makedirs(all_png_path)
    os_makedirs(all_qrcode_path)

    # 扫描文件夹
    folder_list = os.listdir(root_path)
    for folder_i in folder_list:
        print("\n--Label:", folder_i)
        img_list = os.listdir(opj(root_path, folder_i))

        # 遍历文件夹内所有图片
        i = -1
        for img_i in tqdm.tqdm(img_list):

            img_array = cv2.imread(opj(root_path, folder_i, img_i), -1)

            # 识别二维码位置
            img_rect, img_warped, img_qrcode = QRCode_detection(img_array)
            # 如果不存在二维码
            if img_rect is None:
                print("Warning: there are no QRCode in the image:", opj(root_path, folder_i, img_i))
                continue
            else:  # 存在二维码
                # 由于拍摄的图像格式不一，所以读取图片后将图片统一为 png 格式并保存至新的文件夹
                i += 1
                new_folder_i = opj(all_png_path, folder_i)
                os_makedirs(new_folder_i)
                new_img_i = folder_i + "_" + str(i) + ".png"
                if not os.path.exists(opj(new_folder_i, new_img_i)):
                    cv2.imwrite(opj(new_folder_i, new_img_i), img_array)

                # 保存裁剪旋转调整方向后的二维码图片
                qrcode_folder_i = opj(all_qrcode_path, folder_i)
                os_makedirs(qrcode_folder_i)
                if not os.path.exists(opj(qrcode_folder_i, new_img_i)):
                    cv2.imwrite(opj(qrcode_folder_i, new_img_i), img_qrcode)
