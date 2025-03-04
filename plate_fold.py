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
    camera_Principles.init_jr()

    camera_Principles.root_path = root_path

    result_path = root_path + '_result_fold'
    result_path_label = root_path + '_label'
    del_path = result_path+ '_del'
    for path in [result_path,result_path_label,del_path]:
        if not os.path.exists(path):
            os.mkdir(path)
    del_path = result_path+ '_del'
    del_imgs = os.listdir(del_path)
    Vpath_aim = os.path.join(result_path, f'result_ID{camera_Principles.ID}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # video = cv2.VideoWriter(Vpath_aim, fourcc, 0.5, (self.wPixel_real, self.hPixel_real))
    video = None
    # camera_Principles = Camera_Principles()
    # 获取文件和目录列表
    files_and_dirs = os.listdir(root_path)
    # 使用sorted函数按照文件名排序
    sorted_files_and_dirs = sorted(files_and_dirs)
    fps = 7#15

    if True:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5#0.5
        font_thickness = 2

    # 输出排序后的列表
    for name in sorted_files_and_dirs:
        # for index in range(1,3000):
        # name = f'IDdhz33_T00_P145_index{index}.jpg'
        if '_predict' in name:
            continue
        elif '.mp4' in name:
            continue
        elif not str(camera_Principles.ID) in name:
            continue
        # elif 'index902' not in name:
        #     continue
        print('name ', name)

        # match = re.search(r'ID(\d+)_T(\d{2})_P(\d+).*.jpg', name)
        # match = re.search(r'ID([a-zA-Z]+\d+)_T(\d{2})_P(\d+).*.jpg', name)
        # match = re.search(r'ID(\w+)_T(\d{2})_P(\d+).*.jpg', name)
        if 'index' in name:
            match = re.search(r'ID(\w+)_T(\d+)_P(\d+)_index(\d+).*.jpg', name)
            index = int(match.group(4))
            if index<496:
                continue
            if index>1796:
                continue
            elif name.split('.')[0]+'_img_result.png' in del_imgs:
                continue
        else:
            match = re.search(r'ID(\w+)_T(\d+)_P(\d+).*.jpg', name)

        ID = str(match.group(1))
        Tilt = int(match.group(2))
        Pan = int(match.group(3))
        
        Pan = Pan - 180

        # Pan = -55
        # Tilt = 10#0

        if str(camera_Principles.ID) != str(ID):
            continue
        # elif index <496:
        #     continue

        # elif abs(abs(Pan) - 90 )>10:
        #     continue
        # elif abs(Pan) > 25:
        #     continue
        # elif abs(Pan) < 50 or abs(Pan) > 130:
        #     continue


        image1 = cv2.imread(os.path.join(root_path, name))
        img_ori_cv_ = image1.copy()
        result_img_plate = np.zeros_like(image1)
        camera_Principles.hPixel_real, camera_Principles.wPixel_real, _ = image1.shape
            # camera_Principles.hPixel_real, camera_Principles.wPixel_real = 960,1280

        if video is None:
            video = cv2.VideoWriter(Vpath_aim, fourcc, fps, (camera_Principles.wPixel_real, camera_Principles.hPixel_real))
        if camera_Principles.flag_test:
            try:
                np_predict = np.load(os.path.join(root_path, name.split('.')[0] + '_predict_solo.npy'))
                # 从Pickle文件读取字典
                with open(os.path.join(root_path, name.split('.')[0] + '_predict_solo_all_class_dict.pkl'), 'rb') as f:
                    all_class_dict = pickle.load(f)

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
        points_plate_fold_ori = points_plate_fold.copy()
        # img_mark,plates_dict = camera_Principles.mark_4point(points_4d)
        result_img_plate,plates_dict = camera_Principles.mark_4point(points_4d,result_img_plate,flag_only_plates_dict=True)

        plates_id_arr = np.zeros_like(np_predict, dtype=np.int32)

        plates_border2 = {}
        
        for key, value in plates_dict.items():
            # print(f"Key: {key}, Value: {value}")
            value_ = np.array(value.copy(),dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(plates_id_arr, [value_], key+50)

            # 转换为 n*2 的 numpy 数组
            points = np.array(value)

            # 计算 x 坐标的最大值和最小值
            x_max = np.max(points[:, 0])
            x_min = np.min(points[:, 0])

            plates_border2[key] = [x_min,x_max]


        print('plates_border2',plates_border2)
        # plates_border = {}
        # for key, value in plates_border2.items():

        
        # print('np_predict',np_predict)
        print('points_plate_fold',len(points_plate_fold))

        plate_fold_arr = np.zeros_like(np_predict, dtype=int)

        if len(points_plate_fold)>0:
            if abs(Pan)<20:
                B_point = None if points_plate_fold[-1][0]>=camera_Principles.wPixel_real else [camera_Principles.wPixel_real,points_plate_fold[-1][1]]
                C_point = [camera_Principles.wPixel_real, camera_Principles.hPixel_real]
                D_point = [0,camera_Principles.hPixel_real]
                A_point = None if points_plate_fold[-1][0]<=0 else [0,points_plate_fold[-1][1]]
            elif Pan < 0:
                B_point = None if points_plate_fold[-1][0]>=camera_Principles.wPixel_real else [camera_Principles.wPixel_real,points_plate_fold[-1][1]]
                C_point = [camera_Principles.wPixel_real, camera_Principles.hPixel_real]
                D_point = [points_plate_fold[0][0],camera_Principles.hPixel_real]
                A_point = None
            else:
                B_point = None
                C_point = [points_plate_fold[-1][0], camera_Principles.hPixel_real]
                D_point = [0,camera_Principles.hPixel_real]
                A_point = None if points_plate_fold[-1][0]<=0 else [0,points_plate_fold[-1][1]]
    
            for point_ in [B_point,C_point,D_point,A_point]:
                if point_ is not None:
                    points_plate_fold.append(point_)

            points_plate_fold = np.array(points_plate_fold, dtype=np.int32)
            # 因为 cv2.fillPoly 需要输入的点数组是一个形状为 (n, 1, 2) 的数组，格式需要稍微调整
            points_plate_fold = points_plate_fold.reshape((-1, 1, 2))

            # 使用 cv2.fillPoly 填充这个多边形，255表示填充颜色（白色）
            cv2.fillPoly(plate_fold_arr, [points_plate_fold], 100)

        # cv2.imwrite(os.path.join(result_path , name.split('.')[0] + '_plate_fold_arr.png'), plate_fold_arr*2)

        # np.save('points_plate_fold.npy',[points_4d,points_plate_fold,points_line,points_boundary])

        plate_fold_arr_overlap = plate_fold_arr + np_predict

        plate_fold_arr_overlap[plate_fold_arr_overlap >100] = 255

        # cv2.imwrite(os.path.join(result_path , name.split('.')[0] + '_plate_fold_arr_overlap.png'), plate_fold_arr_overlap)

        flag_plate = '所有护帮板折叠'
        un_fold_plate_id = []
        texts = []
        test_xys = []
        if 'plates' in all_class_dict.keys():
            # for index,plate in enumerate(all_class_dict['plates']):
            #     # 转换为 n*2 的形状
            #     plate_reshaped = plate.copy().reshape(-1, 2)
            #     # 计算 x 坐标平均值
            #     x_mean = np.mean(plate_reshaped[:, 0])
            #     # x_mean = 
            #     if len(all_class_dict['plates'])>2:
            #         pass

            if True:
                if len(all_class_dict['plates'])>1:
                    # 提取点集
                    plates = all_class_dict['plates']

                    # 定义排序逻辑
                    if Pan>=0:
                        sorted_plates = sorted(plates, key=lambda plate: np.mean(plate[:, 0, 0]))  # 计算每个点集的 x 均值
                    else:
                        sorted_plates = sorted(plates, key=lambda plate: np.mean(plate[:, 0, 0]),reverse=True)  # 计算每个点集的 x 均值

                    # 将排序结果更新回字典
                    all_class_dict['plates'] = sorted_plates

                    # 打印排序结果
                    for i, plate in enumerate(sorted_plates):
                        center_x = np.mean(plate[:, 0, 0])  # 中心点 x
                        center_y = np.mean(plate[:, 0, 1])  # 中心点 y
                        print(f"区域 {i + 1} 中心点: ({center_x:.2f}, {center_y:.2f})")
                        # print(plate)

            for index,plate in enumerate(all_class_dict['plates']):
                plate_arr = np.zeros_like(np_predict, dtype=int)
                cv2.fillPoly(plate_arr, [plate], 1)
                # if sum(plate_arr)
                if plate_arr.sum()/camera_Principles.wPixel_real/camera_Principles.hPixel_real<0.01:
                    print('面积太小')
                    continue
                else:
                    plate_arr = plate_arr*100 + plate_fold_arr

                    # cv2.imwrite(os.path.join(result_path , name.split('.')[0] + f'_plate_arr{index}.png'), plate_arr)
                    if plate_arr.max() == 200:
                        flag_plate = '有护帮板未折叠'

                        # cv2.imwrite(os.path.join(result_path , name.split('.')[0] + '_plates_id_arr.png'),(plates_id_arr-40)*20)

                        plates_id_arr_ = plates_id_arr.copy()
                        plates_id_arr_[plate_arr!=200]=0
                        # plates_id_arr_.max()
                        # 去除零元素
                        non_zero_arr = plates_id_arr_[plates_id_arr_ != 0]
                        # 如果存在非零元素，继续计算
                        if non_zero_arr.size > 0:
                            # 使用 np.unique() 统计各值出现的次数
                            unique_values, counts = np.unique(plates_id_arr_[plates_id_arr_ != 0], return_counts=True)
                            if False:
                                # 找到出现次数最多的非零元素
                                max_count_index = np.argmax(counts)
                                most_frequent_value = unique_values[max_count_index]
                                most_frequent_count = counts[max_count_index]

                                # 打印结果
                                print(f"出现最多的非零元素是: {most_frequent_value}, 出现次数: {most_frequent_count}")
                                plate_id = most_frequent_value-50
                            else:

                                # 转换为 n*2 的形状
                                plate_reshaped = plate.copy().reshape(-1, 2)

                                # 计算 x 坐标的最大值和最小值
                                x_max = np.max(plate_reshaped[:, 0])
                                x_min = np.min(plate_reshaped[:, 0])
                                
                                if Pan<0:
                                    for i in range(10,-11,-1):
                                        if i in plates_border2.keys():
                                            if x_max>=plates_border2[i][0]:
                                                plate_id = i
                                                break
                                else:
                                    for i in range(-10,11,1):
                                        if i in plates_border2.keys():
                                            if x_min<=plates_border2[i][1]:
                                                plate_id = i
                                                break
                            if plate_id in un_fold_plate_id:
                                if Pan<0:
                                    plate_id -=1
                                else:
                                    plate_id +=1

                            print(plate_id,'未折叠')

                        
                            un_fold_plate_id.append(plate_id)
                            
                            # cv2.fillPoly(result_img_plate, [plate], (0,0,255))
                            # cv2.fillPoly(result_img_plate, [plate], camera_Principles.colour_dict27[5])
                            cv2.fillPoly(result_img_plate, [plate], (0,0,100))

                            numbers = int(''.join(char for char in ID if char.isdigit()))
                            # print(numbers)  # 输出：33

                            text = f'#{str(numbers+plate_id)}Warn!!'
                            # 获取文本的宽度和高度
                            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale,font_thickness)

                            # text_width = min(max(0+100,text_width),camera_Principles.wPixel_real-100)
                            # text_height = min(max(0+20, text_height),camera_Principles.hPixel_real-20)

                            # # 计算行均值
                            # row_means = np.mean(plate, axis=0)  # axis=1表示沿着行计算均值
                            # print("行均值:", row_means)  # 每行的均值都是1.0

                            # # 计算列均值
                            # col_means = np.mean(plate, axis=0)  # axis=0表示沿着列计算均值
                            # print("列均值:", col_means)  # 每列的均值都是1.0

                            row_means,col_means = np.mean(plate, axis=0)[0]

                            # 计算文本的左下角坐标（OpenCV的putText函数要求的）
                            # text_x = plate[0][0][0] - text_width // 2
                            # text_y = plate[0][0][1] + text_height // 2

                            text_x = row_means - text_width // 2
                            text_y = col_means + text_height // 2

                            # cv2.putText(result_img_plate, text, (text_x, text_y), font, font_scale, camera_Principles.colour_dict27[6],font_thickness)
                            # cv2.putText(result_img_plate, text, (text_x, text_y), font, font_scale, (0,0,255),font_thickness)
                            # plate
                            
                            texts.append(text)
                            test_xys.append([text_x,text_y])
                            
                        else:
                            print('不属于任何id')

                    else:
                        print('折叠')
                        cv2.fillPoly(result_img_plate, [plate], (0,255,0))
        else:
            flag_plate = '未识别出护帮板'
        print('flag_plate',flag_plate)
        print('un_fold_plate_id',un_fold_plate_id)


        img_result = cv2.addWeighted(img_ori_cv_.copy(), 0.7, result_img_plate, 0.5, 0)
        img_result = camera_Principles.single_mark_lines_palte_fold(points_plate_fold_ori, img_result)

        for i in range(len(texts)):
            text = texts[i]
            text_x, text_y = test_xys[i]
            # cv2.putText(img_result, text, (text_x, text_y), font, font_scale, (0,0,255),font_thickness)

            cv2.putText(img_result, text, (int(text_x), int(text_y)), font, font_scale, camera_Principles.colour_dict8[4],font_thickness)

        
        cv2.imwrite(os.path.join(result_path , name.split('.')[0] + '_img_result.png'), img_result)
        video.write(img_result)

        # # 将对象保存到文件
        # with open('points_plate_fold.pkl', 'wb') as file:
        #     pickle.dump([points_4d,points_plate_fold,points_line,points_boundary], file)


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
        # root_path = r'/data/ins/dataset/test/all'
        # root_path = r'/data/ins/dataset/test/test2'
        # root_path = r'/data/ins/dataset/test/dif
        root_path = r'/data/ins/dataset/Virtual_plate_number/solov2/video_img'
    else:
        # root_path = r'F:\Desktop\move_ptz_img\angle_#68_result'
        # root_path = r'F:\Desktop\move_ptz_img\angle_#68_result2'
        # root_path = r'F:\Desktop\move_ptz_img\angle_#68'
        # root_path = r'F:\Desktop\move_ptz_img\angle_#44'
        # root_path = r'F:\Desktop\move_ptz_img\all'
        # root_path = r'F:\Desktop\move_ptz_img\all_result'
        # root_path = r'D:\dataset\Virtual_plate_number'
        # root_path = r'D:\dataset\Virtual_plate_number\all'
        # root_path = r'D:\dataset\Virtual_plate_number\solov2\all_solo'
        # root_path = r'D:\dataset\Virtual_plate_number\solov2\video'
        # '7-1731029629037-034.h264'
        # root_path = r'D:\dataset\Virtual_plate_number\solov2\video_img'
        # root_path = r'D:\code\win\del\Virtual_plate_number\data'
        root_path = r'D:\dataset\Virtual_plate_number\solov2\all_solo'
        # root_path = r'./data'
    IDs = [44,68]
    # IDs = [68, 44]
    # IDs = [44]
    # IDs = [68]
    # IDs = [68,44,103,118,999]
    # IDs = [103,118,999]
    IDs = ['dhz33',44,68]
    


    if True:
        for ID in IDs:
            # # 创建子进程
            if platform.system() == 'Linux':
                p = Process(target=main, args=(ID, root_path))  # target进程执行的任务, args传参数（元祖）
                p.start()  # 启动进程
            else:
                main(ID, root_path)







