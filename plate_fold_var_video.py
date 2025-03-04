import math
import os
# from loss_jr import Loss_jr
import platform
import re
import time
from multiprocessing import Process

import cv2
import numpy as np
from scipy.optimize import minimize, Bounds
from Camera_Principles_GD import Camera_Principles
import pickle  

def main(ID, root_path):
    flag_minimize_count = True
    # 需要转
    camera_Principles = Camera_Principles(ID, root_path)
    # self.cam_num_transmissionBelts = 4 #12# 30#10#4 #100
    camera_Principles.cam_num_transmissionBelts = 10  # 6#10
    camera_Principles.cam_num_plates = 4  # 0#4
    # camera_Principles.hPixel_real, camera_Principles.wPixel_real = 1000,1000#960,1280

    camera_Principles.hPixel_real, camera_Principles.wPixel_real = 960,1280

    camera_Principles.init_jr()
    camera_Principles.root_path = root_path

    result_path = root_path + '_result'
    result_path_label = root_path + '_label'
    for path in [result_path,result_path_label]:
        if not os.path.exists(path):
            os.mkdir(path)

    Vpath_aim = os.path.join(result_path, f'result_ID{camera_Principles.ID}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # video = cv2.VideoWriter(Vpath_aim, fourcc, 0.5, (self.wPixel_real, self.hPixel_real))
    video = None
    # camera_Principles = Camera_Principles()
    # 获取文件和目录列表
    files_and_dirs = os.listdir(root_path)
    # 使用sorted函数按照文件名排序
    sorted_files_and_dirs = sorted(files_and_dirs)


    # 输出排序后的列表
    for name in sorted_files_and_dirs:
        if '_predict' in name:
            continue
        elif '.mp4' in name:
            continue

        elif not str(camera_Principles.ID) in name:
            continue
        print('name ', name)

        match = re.search(r'ID(\d+)_T(\d{2})_P(\d+).*.jpg', name)
        ID = int(match.group(1))
        Tilt = int(match.group(2))
        Pan = int(match.group(3))
        Pan = Pan - 180

        Pan = -55
        Tilt = 10#0

        if str(camera_Principles.ID) != str(ID):
            continue

        # elif abs(abs(Pan) - 90 )>10:
        #     continue
        # elif abs(Pan) > 25:
        #     continue
        # elif abs(Pan) < 50 or abs(Pan) > 130:
        #     continue

        image1 = cv2.imread(os.path.join(root_path, name))
        # camera_Principles.hPixel_real, camera_Principles.wPixel_real, _ = image1.shape
        
        if video is None:
            video = cv2.VideoWriter(Vpath_aim, fourcc, 0.5, (camera_Principles.wPixel_real, camera_Principles.hPixel_real))
        if camera_Principles.flag_test:
            try:
                np_predict = np.load(os.path.join(root_path, name.split('.')[0] + '_predict_solo.npy'))
            except:
                np_predict = None
                print(os.path.join(root_path, name.split('.')[0] + '_predict_solo.npy'), '不存在')

        # Tilt, Pan, = 15,30
        # Tilt, Pan, = 30, -90

        # 需要转
        # # 注意：这会导致像素值的简单相加，可能导致溢出，你可能需要对结果进行归一化
        # result = cv2.addWeighted(image1, 0.5, img_mark,0.5, 0)
        # if camera_Principles.flag_test and (np_predict is not None):  # Dan_pat
        np_predict[np_predict != camera_Principles.list_class.index('plates')] = 0

        points_4d,points_plate_fold,points_line,points_boundary = camera_Principles.count_no_img(Tilt, Pan,camera_Principles.wPixel_real, camera_Principles.hPixel_real)


        # print('np_predict',np_predict)
        print('points_plate_fold',len(points_plate_fold),points_plate_fold)
        # np.save('points_plate_fold.npy',[points_4d,points_plate_fold,points_line,points_boundary])


        # 将对象保存到文件
        with open('points_plate_fold.pkl', 'wb') as file:
            pickle.dump([points_4d,points_plate_fold,points_line,points_boundary], file)


        # if camera_Principles.flag_test or True:
        #     points_4d,points_plate_fold,points_line,points_boundary = camera_Principles.count_no_img(Tilt, Pan,camera_Principles.wPixel_real, camera_Principles.hPixel_real)
        #     # result = img_mark
        #     # 保存结果图片
        #     path_r = os.path.join(result_path, name.split('.')[0] + '_T' + str(Tilt) + '_P' + str(Pan))
        #     # cv2.imwrite(path_r + '.jpg', result)
        #     if np_predict is not None and False:
        #         cv2.imwrite(path_r + '_img_mark_label.png', img_mark_label)
        #         cv2.imwrite(path_r + '_count_like_label_np.png', count_like_label_np * 30)
        #         cv2.imwrite(path_r + '_np_predict.png', np_predict * 30)

        #         # 使用 == 运算符比较两个数组

        #         # 检查形状是否匹配
        #         if count_like_label_np.shape == np_predict.shape:
        #             # equal_element = count_like_label_np == np_predict
        #             pass
        #         else:
        #             print("Arrays do not have the same shape.")
        #         equal_element = count_like_label_np == np_predict
        #         cv2.imwrite(os.path.join(path_r + '_np_equal_element.png'), equal_element * 255)
        #     # video.write(result)
        # else:
        #     camera_Principles.count(Tilt, Pan, image1)


    del camera_Principles




if __name__ == '__main__':
    if platform.system() == 'Linux':
        # input = r'/data/ins/dataset/test/img0705'
        root_path = r'/data/ins/dataset/test/all'
        # root_path = r'/data/ins/dataset/test/test2'
        # root_path = r'/data/ins/dataset/test/dif'
    else:
        # root_path = r'F:\Desktop\move_ptz_img\angle_#68_result'
        # root_path = r'F:\Desktop\move_ptz_img\angle_#68_result2'
        # root_path = r'F:\Desktop\move_ptz_img\angle_#68'
        # root_path = r'F:\Desktop\move_ptz_img\angle_#44'
        # root_path = r'F:\Desktop\move_ptz_img\all'
        # root_path = r'F:\Desktop\move_ptz_img\all_result'
        # root_path = r'D:\dataset\Virtual_plate_number'
        # root_path = r'D:\dataset\Virtual_plate_number\all'
        root_path = r'D:\dataset\Virtual_plate_number\solov2\all_solo'
        # root_path = r'D:\dataset\Virtual_plate_number\solov2\video'
        # '7-1731029629037-034.h264'
    # IDs = [44,68]
    # IDs = [68, 44]
    IDs = [44]
    # IDs = [68]
    # IDs = [68,44,103,118,999]
    # IDs = [103,118,999]


    if True:
        for ID in IDs:
            # # 创建子进程
            if False or platform.system() == 'Linux':
                p = Process(target=main, args=(ID, root_path))  # target进程执行的任务, args传参数（元祖）
                p.start()  # 启动进程
            else:
                main(ID, root_path)










