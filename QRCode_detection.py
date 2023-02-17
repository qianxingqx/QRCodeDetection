# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:18:13 2023
Description: 对输入的图片进行预处理，提取出图片中的二维码，并裁剪得到二维码区域，然后旋转至正确方向
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
    cv2.namedWindow(img_name, 0)  # 代表窗口可伸缩，否则 imshow 会显示不全
    cv2.imshow(img_name, img_array)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def os_makedirs(path):
    """ 不存在则创建文件夹
    """
    if not os.path.exists(path):
        os.makedirs(path)


def QRCode_detection(img_array):
    """ 检测图片中的二维码，只允许检测单个
    Args:
        img_array
    Returns:
        img_rect, img_warped, img_cropped
    """
    # 转换为灰度图像
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # 找到图像中的所有二维码
    # 可改为：test=pyzbar.decode(frame,ymbols=[pyzbar.ZBarSymbol.QRCODE])
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
        width = rect[2]
        height = rect[3]
        src_pts = np.float32([points[0][0], points[1][0], points[2][0], points[3][0]])
        dst_pts = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        img_warped = cv2.warpPerspective(img_array, M, (width, height))

        # 将二维码位置标记在原图上
        img_rect = img_array.copy()
        line_color = (0, 0, 255)
        line_thickness = 3
        cv2.line(img_rect, (int(src_pts[0][0]), int(src_pts[0][1])), (int(src_pts[1][0]), int(src_pts[1][1])), color=line_color, thickness=line_thickness)
        cv2.line(img_rect, (int(src_pts[1][0]), int(src_pts[1][1])), (int(src_pts[2][0]), int(src_pts[2][1])), color=line_color, thickness=line_thickness)
        cv2.line(img_rect, (int(src_pts[2][0]), int(src_pts[2][1])), (int(src_pts[3][0]), int(src_pts[3][1])), color=line_color, thickness=line_thickness)
        cv2.line(img_rect, (int(src_pts[3][0]), int(src_pts[3][1])), (int(src_pts[0][0]), int(src_pts[0][1])), color=line_color, thickness=line_thickness)

        # 将二维码统一旋转至 UP 方向
        # LEFT |-, DOWN |_, RIGHT _|, UP -|
        img_warped_gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
        img_warped_decoded_objs = decode(img_warped_gray)
        if len(img_warped_decoded_objs) == 0:
            # print("No QR code found in the image.")
            return None, None, None
        img_warped_obj = img_warped_decoded_objs[0]
        # print(img_warped_obj.orientation)
        if img_warped_obj.orientation == "LEFT":
            img_rotate = np.rot90(img_warped, k=-3)  # 顺时针旋转 270
        elif img_warped_obj.orientation == "UP":
            img_rotate = np.rot90(img_warped, k=-2)  # 顺时针旋转 180
        elif img_warped_obj.orientation == "RIGHT":
            img_rotate = np.rot90(img_warped, k=-1)  # 顺时针旋转 90
        else:
            img_rotate = img_warped.copy()

        # # 将二维码裁剪成正方形
        # if width > height:
        #     img_cropped = img_rotate[:, (width-height)//2:(width+height)//2]
        # else:
        #     img_cropped = img_rotate[(height-width)//2:(height+width)//2, :]

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
        print("\n--Tag:", folder_i)
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
