import math
import cv2
import re
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial import ConvexHull
# from loss_jr import Loss_jr
import platform
from scipy.optimize import minimize, Bounds, LinearConstraint  
import numpy as np
from multiprocessing import Pool
import os, time, random
from multiprocessing import Process

class Camera_Principles():
    def __init__(self,ID=None,root_path=None):
        # 设置
        self.flag_Move_first = True
        self.list_class  = ['background', 'plates', 'transmissionBelt', 'track', 'cableTrough', 'groovedSide', 'coal', 'roller']
        self.np_predicts = None
        self.count_like_label_nps = None
        self.ID = str(ID)
        self.root_path = root_path
        self.name_initial_guess = f"initial_guess_ID{self.ID}.npy"
        self.flag_calculation_depends = []
        self.time_begin = time.time()
        self.flag_plate_UP = False
        self.flag_test = True  #调试模式开关  如果不想画图，不想测试，该值为False,  
        self.wPixel_referenc = 704
        self.hPixel_referenc = 576
        self.flag_vstack = True


        if False:#51
            self.fx, self.fy = [468.61,680.34]#[593.74,658.35]#[542.9,629.46]
            self.dist = np.array([-4.24103449e-01,2.11013612e-01,-1.43886286e-03,-3.49882761e-05,-5.33531165e-02],dtype=np.float32)
        else:#all
            self.fx, self.fy = [468.64,680.222]
            self.dist = np.array([-4.23709796e-01,2.23283964e-01,-6.57576014e-04, -4.92667899e-05,-6.80752617e-02],dtype=np.float32)

        self.k1,self.k2,self.p1,self.p2,self.k3 = self.dist
         # 不需要转
        if os.path.exists(self.name_initial_guess) and self.flag_test:
            self.d_level, self.camera_h_word, self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support  = np.load(self.name_initial_guess,allow_pickle=True)
        # 需要转
        else:
            # raise 'os.path.exists(self.name_initial_guess) and self.flag_test'
            # self.d_level, self.camera_h_word, self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support  = 3390,2500,1872,12,-2.6
            # self.d_level, self.camera_h_word, self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support  = 3390,2500,1872,0,0
            self.d_level, self.camera_h_word, self.transmissionBelt_Coal_wall1, self._T_constant_camera, self._T_constant_support = 3390, 2500, 1872, 0,0
        if False and platform.system() == 'Windows' :#temp ＩＤ４４　 画图transmissionBelt_Coal
            if self.ID == '44' :
                # self._T_constant_support = -10
                # self._T_constant_camera -= 8
                self._T_constant_support = -self._T_constant_camera -8
                # self.d_level += 1000
                # self.transmissionBelt_Coal_wall1 +=
                # self.camera_h_word -= 800
                self.transmissionBelt_Coal_wall1 -= 100
                # else:
            #     break


        self.error_function_N0 = 0
        self.flag_print_detal = False

        # 设备型号：DS-2DC2D40IW-DE3
        # 1、一些参数
        # 1.1、无误差值
        self.w_plate_word = 1400 #self.w_plate_word 护帮板实际宽度，
        self.h_plate_word = 1340 # self.h_plate_wor 护帮板实际高度
        self.Unwanted_h_plate_word = 300
        self.w_transmissionBelt_word = 800#1000 #800#1500#800#1500 #17000 #800
        self.f = 2.8
        self.HFOV_half = 97 / 2  # —— 水平视场角（Horizontal Field of View）的一半
        self.transmissionBelt_Coal_wall0 = 0
        self.y_higher_plate = 0 #-100#-300

        self.transmissionBelt_Coal_wall3 = 4.5e3 #5.5e3

        self.cam_num_plates = 0 #8#10#13 # 10#15#20 #20
        self.cam_num_transmissionBelts = 0 #10 #10#16#10#13 # 10#15#20 #20
        # boundaries_id = [0,1,2,3,4,6,7,8,9,12,16,25,50]
        boundaries_id = [0, 1, 2, 3, 4, 6, 7, 8, 9, 12]
        boundaries_id = boundaries_id + [-x for x in boundaries_id]
        self.boundaries_id = sorted(set(boundaries_id))
        self.Installation_offset = 0 #-800 # 相机安装的偏移位置，从支架视角看，以左为负，右为正，安装到支架中间，该值为0。单位：mm。


        self.wPixel_real = self.wPixel_referenc #2000 #704 # 实际像素值
        self.hPixel_real = self.hPixel_referenc #2000 # 576 # 实际像素值


        # 颜色
        if True:
            self.colour_dict8 = {
                0: (0, 0, 0),  # 黑色
                1: (255, 0, 0),  # 蓝色
                2: (0, 0, 255),  # 红色
                3: (0, 255, 0),  # 绿色
                4: (0, 255, 255),  # 黄色
                5: (255, 0, 255),  # 洋红色
                6: (255, 255, 0),  # 青色
                7: (255, 255, 255)  # 白色
            }

            self.colour_dict27 = {
                0: (0, 0, 0),  # 黑色
                1: (0, 0, 128),  # 深蓝
                2: (0, 0, 255),  # 蓝色
                3: (0, 128, 0),  # 深绿色
                4: (128, 128, 0),  # 深青色

                # 红色系列
                5: (0, 0, 128),  # 深红色
                6: (0, 0, 255),  # 红色
                7: (128, 128, 255),  # 浅粉色

                # 黄色系列
                8: (0, 255, 255),  # 黄色
                9: (128, 255, 255),  # 浅黄色

                # 绿色系列
                10: (0, 255, 0),  # 绿色
                11: (0, 255, 128),  # 浅绿色
                12: (128, 255, 128),  # 浅青色

                # 青色系列
                13: (255, 255, 0),  # 青色
                14: (255, 255, 128),  # 浅青色

                # 紫色系列
                15: (128, 0, 128),  # 深紫色
                16: (255, 0, 255),  # 洋红色

                # 白色系列
                17: (255, 255, 255),  # 白色
                18: (224, 224, 224),  # 浅灰色

                # 灰色系列
                19: (128, 128, 128),  # 中灰色
                20: (64, 64, 64),  # 深灰色

                # 橙色系列
                21: (0, 128, 255),  # 橙色
                22: (0, 160, 255),  # 浅橙色

                # 棕色系列
                23: (0, 64, 128),  # 深棕色
                24: (0, 80, 160),  # 浅棕色

                # 粉色系列
                25: (203, 192, 255),  # 粉色
                26: (238, 220, 255),  # 浅粉色
            }

            self.colour_dict16 = {
                0: (0, 0, 0),  # 黑色
                1: (255, 0, 0),  # 蓝色 #0000FF
                2: (149, 140, 205),  # 粉红色 #CD8C95
                3: (173, 107, 67),  # 深蓝色 #436BAD
                4: (0, 173, 205),  # 金色 #CDAD00
                5: (137, 244, 4),  # 绿色 #04F489
                6: (154, 1, 254),  # 洋红色 #FE019A
                7: (12, 71, 6),  # 深绿色 #06470C
                8: (42, 222, 97),  # 浅绿色 #61DE2A
                9: (95, 248, 203),  # 浅黄色 #CBF85F
                10: (255, 187, 255),  # 浅粉色 #FFBBFF
                11: (212, 255, 127),  # 浅绿色 #7FFFD4
                12: (31, 44, 211),  # 深红色 #D32C1F
                13: (254, 204, 2),  # 青色 #02CCFE
                14: (250, 0, 153),  # 紫色 #9900FA
                15: (81, 20, 93),  # 深紫色 #5D1451
                16: (255, 255, 255),  # 白色
            }

            self.colour_dict = self.colour_dict8
            colour_interval = 100
            colour_wh = int(len(self.colour_dict)**(1/2))+2
            colour_img = np.zeros([colour_wh*colour_interval,colour_wh*colour_interval,3])

            for k in self.colour_dict.keys():
                y = (k // (colour_wh-1) + 1) *colour_interval
                x = (k % (colour_wh-1) +1 )* colour_interval
                # 在图片上画点
                cv2.circle(colour_img, (x, y), 20, self.colour_dict[k], -1)
                # 在图片上添加文本标记序号
                cv2.putText(colour_img, str(k), (x - 30, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colour_dict[k], 1)
            cv2.imwrite('./5.jpg',colour_img)
        
        # self.init_jr()

     # 需要转
    def init_jr(self):
        self.np_predicts = None
        self.count_like_label_nps = None

        self.W_half_pixel = self.wPixel_real/2
        #1.2、有误差值的观测者
        # 9203,68# T00 w191h226
        self.w_plate_pixel = 194#191#200 #205#201 #165
        self.h_plate_pixel = 227#226#250 #272#256 #250
        self.interval_plate_pixel = 46#43#41# (41+31)/2#43


        # 1.3、有误差值的推测值
        # self.transmissionBelt_Coal_wall1 = 1500 #1500#1500#1500


        # 1.3、有误差值的计算值
        if False:
            self.interval_plate_word = (self.w_plate_word / self.w_plate_pixel) * self.interval_plate_pixel
        else:
            self.interval_plate_word = 332

        if False:
            self.W_half_level = self.W_half_pixel * (self.w_plate_word / self.w_plate_pixel)  # 视野在水平方向看到的距离的一半
            # self.d_level = 3000  # 3808#2808
            if self.d_level is None:
                self.d_level = self.W_half_level / self.tan_jr_degrees(self.HFOV_half)
                self.d_level = self.d_level*1.5

            if self.fx is None and False:
                self.fx = (self.w_plate_pixel * self.d_level) / self.w_plate_word
                self.fy = (self.h_plate_pixel * self.d_level) / self.h_plate_word


        # 大了往里,小了往外

        # 2.0 最少需要求解的参数
        '''
        self._T_constant_camera = 6#7#12#10 #5 #10 # -5 #相机向下偏转的角度误差 # 远处往下跑，要调大
        self.d_level 
        self.camera_h_word
        self.transmissionBelt_Coal_wall1
        self.transmissionBelt_Coal_wall2
        '''


        self.transmissionBelt_Coal_wall2 = self.transmissionBelt_Coal_wall1 + self.w_transmissionBelt_word
        if False:#破坏性测试，必须关闭
            self.d_level = self.transmissionBelt_Coal_wall2
        # 单位mm,毫米
        # 1.世界坐标系

        self.plate_Word = [
            [-self.w_plate_word / 2, self.Unwanted_h_plate_word, 0],
            [-self.w_plate_word / 4, self.Unwanted_h_plate_word, 0],
            [0, self.Unwanted_h_plate_word, 0],
            [self.w_plate_word / 4, self.Unwanted_h_plate_word, 0],
            [self.w_plate_word / 2, self.Unwanted_h_plate_word, 0],


            [self.w_plate_word / 2, self.h_plate_word, 0],
            [self.w_plate_word / 4, self.h_plate_word, 0],
            [0, self.h_plate_word, 0],
            [-self.w_plate_word / 4, self.h_plate_word, 0],
            [-self.w_plate_word / 2, self.h_plate_word, 0],
              ]

        # self.camera_h_word -=1000
        # self.transmissionBelt_Coal_wall1 -=1000
        hh_plate = 200
        h_dm = 200
        Ground_error = 800
        # 墙线 带入误差
        if True:
            self.transmissionBelt_Coal_wall0 = -Ground_error

        self.transmissionBelt_Word = [


            # # 护帮板安全线1
            # [-self.w_plate_word / 2, self.Unwanted_h_plate_word+hh_plate, -self.transmissionBelt_Coal_wall0],
            # [-self.w_plate_word / 4, self.Unwanted_h_plate_word+hh_plate, -self.transmissionBelt_Coal_wall0],
            # [0, self.Unwanted_h_plate_word+hh_plate, -self.transmissionBelt_Coal_wall0],
            # [self.w_plate_word / 4, self.Unwanted_h_plate_word+hh_plate, -self.transmissionBelt_Coal_wall0],
            # [self.w_plate_word / 2, self.Unwanted_h_plate_word+hh_plate, -self.transmissionBelt_Coal_wall0],


            # # 护帮板安全线2

            # [-self.w_plate_word / 2, self.h_plate_word-h_dm, -self.transmissionBelt_Coal_wall0],
            # [-self.w_plate_word / 4, self.h_plate_word-h_dm, -self.transmissionBelt_Coal_wall0],
            # [0, self.h_plate_word-h_dm, -self.transmissionBelt_Coal_wall0],
            # [self.w_plate_word / 4, self.h_plate_word-h_dm, -self.transmissionBelt_Coal_wall0],
            # [self.w_plate_word / 2, self.h_plate_word-h_dm, -self.transmissionBelt_Coal_wall0],

            # 墙线
            [-self.w_plate_word / 2, self.camera_h_word, -self.transmissionBelt_Coal_wall0],
            [-self.w_plate_word / 4, self.camera_h_word, -self.transmissionBelt_Coal_wall0],
            [0, self.camera_h_word, -self.transmissionBelt_Coal_wall0],
            [self.w_plate_word / 4, self.camera_h_word, -self.transmissionBelt_Coal_wall0],
            [self.w_plate_word / 2, self.camera_h_word, -self.transmissionBelt_Coal_wall0],

            # 传送带线1
            [-self.w_plate_word / 2, self.camera_h_word, -self.transmissionBelt_Coal_wall1],
            [-self.w_plate_word / 4, self.camera_h_word, -self.transmissionBelt_Coal_wall1],
            [0, self.camera_h_word, -self.transmissionBelt_Coal_wall1],
            [self.w_plate_word / 4, self.camera_h_word, -self.transmissionBelt_Coal_wall1],
            [self.w_plate_word / 2, self.camera_h_word, -self.transmissionBelt_Coal_wall1],

            # 传送带线2
            [-self.w_plate_word / 2, self.camera_h_word, -self.transmissionBelt_Coal_wall2],
            [-self.w_plate_word / 4, self.camera_h_word, -self.transmissionBelt_Coal_wall2],
            [0, self.camera_h_word, -self.transmissionBelt_Coal_wall2],
            [self.w_plate_word / 4, self.camera_h_word, -self.transmissionBelt_Coal_wall2],
            [self.w_plate_word / 2, self.camera_h_word, -self.transmissionBelt_Coal_wall2],


            # 大脚趾线
            [-self.w_plate_word / 2, self.camera_h_word, -self.transmissionBelt_Coal_wall3],
            [-self.w_plate_word / 4, self.camera_h_word, -self.transmissionBelt_Coal_wall3],
            [0, self.camera_h_word, -self.transmissionBelt_Coal_wall3],
            [self.w_plate_word / 4, self.camera_h_word, -self.transmissionBelt_Coal_wall3],
            [self.w_plate_word / 2, self.camera_h_word, -self.transmissionBelt_Coal_wall3],
            ]


        self.boundary_Word = [
            # # 竖直在煤壁上的一条线
            # [0, 0, 0],
            # [0, self.Unwanted_h_plate_word, 0],
            # [0, self.h_plate_word, 0],
            # [0, (self.h_plate_word + self.camera_h_word)/2, 0],
            # [0, self.camera_h_word, 0],


            #地面上的线
            [0, self.camera_h_word, Ground_error],
            [0, self.camera_h_word, -self.transmissionBelt_Coal_wall0],
            [0, self.camera_h_word, -self.transmissionBelt_Coal_wall1],
            [0, self.camera_h_word, -self.transmissionBelt_Coal_wall2],
            [0, self.camera_h_word, -self.d_level],
            [0, self.camera_h_word, -4e3],
            [0, self.camera_h_word, -4.5e3],
            # [0, self.camera_h_word, -5e3],
            # [0, self.camera_h_word, -6e3],
            # [0, self.camera_h_word, -7e3],
        ]

        if False:
            self.boundary_Word = [
                #竖直在煤壁上的一条线
                [0, 0, 0],
                [0, self.Unwanted_h_plate_word, 0],
                [0, self.h_plate_word, 0],
                [0, (self.h_plate_word + self.camera_h_word)/2, 0],
                [0, self.camera_h_word, 0],

                #地面上的线

                [0, self.camera_h_word, 0],
                [0, self.camera_h_word, -self.transmissionBelt_Coal_wall1],
                [0, self.camera_h_word, -self.transmissionBelt_Coal_wall2],
                [0, self.camera_h_word, -self.d_level],
                [0, self.camera_h_word, -4e3],
                [0, self.camera_h_word, -5e3],
                [0, self.camera_h_word, -6e3],
                [0, self.camera_h_word, -7e3],
            ]
        # self.d_level = self.d_level*10

        self.plates_Word = []
        # self.plates_Word_calibration = []
        self.transmissionBelts_Word = []
        self.boundaries_Word = []
        self.plate_center_distance = self.w_plate_word + self.interval_plate_word # 

        Z_Coal_wall_word  = self.d_level if self.flag_Move_first else 0

        for i in range(-self.cam_num_plates, self.cam_num_plates + 1, 1):
            for wi in self.plate_Word:
                self.plates_Word.append([wi[0] + (self.w_plate_word + self.interval_plate_word) * i, wi[1] + self.y_higher_plate, wi[2]+Z_Coal_wall_word])

        for i in range(-self.cam_num_transmissionBelts, self.cam_num_transmissionBelts + 1, 1):
            for wi in self.transmissionBelt_Word:
                self.transmissionBelts_Word.append([wi[0] + (self.w_plate_word + self.interval_plate_word) * i, wi[1] + self.y_higher_plate, wi[2]+Z_Coal_wall_word])



        for i in self.boundaries_id:
            for wi in self.boundary_Word:
                # self.boundaries_Word.append([wi[0] + (self.w_plate_word + self.interval_plate_word) * i,wi[1] + self.y_higher_plate, wi[2]+Z_Coal_wall_word])
                self.boundaries_Word.append([wi[0]-self.w_plate_word / 2 + (self.w_plate_word + self.interval_plate_word) * i,wi[1] + self.y_higher_plate, wi[2]+Z_Coal_wall_word])
            for wi in self.boundary_Word:
                self.boundaries_Word.append([wi[0]+self.w_plate_word / 2 + (self.w_plate_word + self.interval_plate_word) * i,wi[1] + self.y_higher_plate, wi[2]+Z_Coal_wall_word])

        
        # self.all_points_Word = self.plates_Word + self.transmissionBelts_Word + self.boundaries_Word

        self.plates_Word = [[-self.Installation_offset+p[0],0+p[1],0+p[2]] for p in self.plates_Word]
        self.transmissionBelts_Word = [[-self.Installation_offset+p[0],0+p[1],0+p[2]] for p in self.transmissionBelts_Word]
        self.boundaries_Word = [[-self.Installation_offset+p[0],0+p[1],0+p[2]] for p in self.boundaries_Word]


        self.K_count()
        self.print_jr('self.flag_calculation_depends,self.d_level, self.camera_h_word, self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support',
                      self.flag_calculation_depends,self.d_level, self.camera_h_word, self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support,)



    # 需要转
    def World_pixel(self,Worldsx3,Rt,K,):
        # 获取数组的形状
        Worlds_nx3 = np.array(Worldsx3)
        # 获取数组的形状
        rows, cols = Worlds_nx3.shape
        # 创建一个新的Nx4的数组，并初始化为1
        Worlds_nx4 = np.ones((rows, cols + 1))
        # 将原始数组的值复制到新的数组中
        Worlds_nx4[:, :3] = Worlds_nx3
        Worlds_4xn = Worlds_nx4.T
        # 1.将世界坐标点转换到相机坐标系 (3x4) * (4xN) = (3xN)
        P_C = Rt @ Worlds_4xn
        # 找出最后一列元素小于0的元素索引
        space_ok = P_C[-1] > 1000

        # 计算归一化图像平面坐标 (3xN)
        P_N = P_C[:3, :] / P_C[2, :]
        # 2.使用相机内参矩阵将归一化图像平面坐标转换为像素坐标 (3x3) * (3xN) = (3xN)
        P_pixel = K @ P_N

        # # 提取像素坐标 u, v
        # u = P_pixel[0, :]
        # v = P_pixel[1, :]
        P_pixel_f = P_pixel.T[:,:2]
        P_pixel_r = P_pixel_f

        k = 0.33#0.33 #0.35 #0.4 #0.4#0.2#0.3 #0.5就乱了

        # 创建一个布尔数组来标识满足条件的行
        satisfying_condition_f = (P_pixel_f[:, 0] >= -self.wPixel_referenc*k) & (P_pixel_f[:, 0] <= self.wPixel_referenc*(1+k)) & (P_pixel_f[:, 1] >= -self.hPixel_referenc*k) & (P_pixel_f[:, 1] <=self.hPixel_referenc*(1+k))
        

        # # 取反以标识不满足条件的行
        # not_satisfying_condition = ~satisfying_condition
        #
        # # 使用 np.where 来找到不满足条件的行的索引
        # indices = np.where(not_satisfying_condition)[0]

        ok_index = space_ok & satisfying_condition_f

        if False:#和另外一种方法计算不一样，这个有问题
            # 4. 校正畸变
            # undistorted_pixel_coords = undistort_points(pixel_coords, K, dist_coeffs)
            pixel_coords = np.expand_dims(P_pixel_f, axis=1)  # 增加一个维度以符合OpenCV格式
            undistorted_points = cv2.undistortPoints(pixel_coords, self.K, self.dist, None, self.K).squeeze()
            P_pixel_r = undistorted_points

            
        if True:
            # 将旋转矩阵转换为旋转向量
            R = Rt[:,:3]
            rvec, _ = cv2.Rodrigues(R)
            # 使用cv2.projectPoints将三维点转换为图像平面的像素坐标
            # image_points, _ = cv2.projectPoints(object_points, rvec, tvec, mtx, dist)
            image_points, _ = cv2.projectPoints(Worlds_nx3, rvec, self.t, self.K, self.dist)
            '''
            object_points：世界坐标系中的三维点。
            rvec：旋转向量。
            tvec：平移向量。
            mtx：内参矩阵。
            dist：畸变系数。
            返回值image_points为像素坐标系中的二维点。
            '''
            # 将像素坐标转换为二维坐标点
            image_points_2d = image_points[:, 0, :2]
            # P_pixel_r = image_points.squeeze()

            P_pixel_r = image_points_2d

        satisfying_condition_r = (P_pixel_r[:, 0] >= -self.wPixel_referenc * k) & (P_pixel_r[:, 0] <= self.wPixel_referenc * (1 + k)) & (P_pixel_r[:, 1] >= -self.hPixel_referenc * k) & (P_pixel_r[:, 1] <=self.hPixel_referenc * (1 + k))
        ok_index = ok_index & satisfying_condition_r


        P_pixel_r = self.resolution_conversion(P_pixel_r)
        P_pixel_r = np.nan_to_num(P_pixel_r, copy=False, nan=-32768, posinf=-32768, neginf=-32768)

        P_pixel_r = np.round(P_pixel_r).astype(np.int16)
        # P_pixel_r[P_pixel_r<0] = np.int16(-32768)
        # P_pixel_r[P_pixel_r > 400] = np.int16(-32768)


        P_pixel_r[~ok_index] = [np.int16(-32768),np.int16(-32768)]

        return P_pixel_r


    def point_cheak(self,point,w_pixel=None,h_pixel=None,k=0.5):
        if np.isnan(point).any():
            return False
        elif np.any(point == np.int16(-32768)):
            return False
        if w_pixel is None:
            return True
        else:
            if -w_pixel*k < point[0] < w_pixel*(1+k) and -h_pixel*k < point[1] < h_pixel*(1+k):
                return True
            else:
                return False
            

    def points_cheak_right_num(self,points,w_pixel=None,h_pixel=None,k=0.5):
        right_num = 0
        for point in points:
            if self.point_cheak(point,w_pixel,h_pixel,k):
                right_num +=1
        return right_num


    def tan_jr_degrees(self, degrees):
        # 将角度转换为弧度
        radians = math.radians(degrees)
        # 计算正切值
        tan_jr_degreesgent_value = math.tan(radians)
        # print(degrees,'tan_jr_degreesgent_value',tan_jr_degreesgent_value)
        return tan_jr_degreesgent_value

    def cos_jr_degrees(self, degrees):
        # 将角度转换为弧度
        radians = math.radians(degrees)
        # 计算余弦
        cos_jr_degreesgent_value = math.cos(radians)
        # print(degrees,'cos_jr_degreesgent_value',cos_jr_degreesgent_value)
        return cos_jr_degreesgent_value

    def mark_lines(self, points_all, img_mark,num_line_each_group = 4):
        # points_num_each_line = int(len(self.transmissionBelt_Word)/num_line_each_group)
        points_num_each_line = 5
        num_line_each_group = int(len(self.transmissionBelt_Word)/points_num_each_line)
        points_num_each_group = num_line_each_group * points_num_each_line

        num_group = int(len(points_all)/points_num_each_group)
        # 按照顺序连接点
        

        lines1 = np.array([points_all[n] for n in range(len(points_all)) if n % points_num_each_group < points_num_each_line and self.point_cheak(points_all[n],self.wPixel_real, self.hPixel_real,0.5)]).astype(np.int32)
        lines2 = np.array([points_all[n] for n in range(len(points_all)) if points_num_each_line<= n % points_num_each_group < points_num_each_line*2 and self.point_cheak(points_all[n], self.wPixel_real, self.hPixel_real, 0.5)]).astype(np.int32)
        lines3 = np.array([points_all[n] for n in range(len(points_all)) if points_num_each_line*2 <= n % points_num_each_group < points_num_each_line*3 and self.point_cheak(points_all[n], self.wPixel_real, self.hPixel_real, 0.5)]).astype(np.int32)
        
        lines4 = np.array([points_all[n] for n in range(len(points_all)) if points_num_each_line*3 <= n % points_num_each_group and self.point_cheak(points_all[n], self.wPixel_real, self.hPixel_real,0.5)]).astype(np.int32)
        # 中间有停顿的画法，能把间断单独显示
        if False:
            for i in range(num_group):
                for j in range(num_line_each_group):
                    # 画和护帮板对应地方传送带、煤层
                    p0 = points_all[i * points_num_each_group + j * points_num_each_line + 0]
                    p1 = points_all[i * points_num_each_group + j * points_num_each_line + 1]
                    p2 = points_all[i * points_num_each_group + j * points_num_each_line + 2]
                    p3 = points_all[i * points_num_each_group + j * points_num_each_line + 3]
                    p4 = points_all[i * points_num_each_group + j * points_num_each_line + 4]

                    list_p = [p0,p1,p2,p3,p4]
                    # list_p = [p0,p1,p2]
                    if True:
                        for pi in range(len(list_p)-1):
                            if self.point_cheak(list_p[pi], self.wPixel_real, self.hPixel_real) and self.point_cheak(list_p[pi+1], self.wPixel_real, self.hPixel_real):
                                cv2.line(img_mark, tuple(list_p[pi]), tuple(list_p[pi+1]), self.colour_dict8[2], 2)
                        if False:

                            if self.point_cheak(p1, self.wPixel_real, self.hPixel_real) and self.point_cheak(p2, self.wPixel_real, self.hPixel_real):
                                # cv2.line(img_mark, tuple(p1), tuple(p2), (255, 255, 0), 2)
                                cv2.line(img_mark, tuple(p1), tuple(p2), self.colour_dict8[2], 2)
                            if self.point_cheak(p2, self.wPixel_real, self.hPixel_real) and self.point_cheak(p3, self.wPixel_real, self.hPixel_real):
                                # cv2.line(img_mark, tuple(p1), tuple(p2), (255, 255, 0), 2)
                                cv2.line(img_mark, tuple(p2), tuple(p3), self.colour_dict8[2], 2)
                        # 画缝隙
                        if i < num_group-1:
                            pp = points_all[(i+1) * points_num_each_group + j * points_num_each_line]
                            if self.point_cheak(p4, self.wPixel_real, self.hPixel_real) and self.point_cheak(pp, self.wPixel_real, self.hPixel_real):
                                # cv2.line(img_mark, tuple(p1), tuple(p2), (255, 255, 0), 2)
                                cv2.line(img_mark, tuple(p4), tuple(pp), self.colour_dict8[3], 2)
        # # 按照顺序连接点  中间没有间断
        if True:
            lines_list = [lines1,lines2,lines3,lines4]
            # colour_list = [self.colour_dict8[0],self.colour_dict8[5],self.colour_dict8[1],self.colour_dict8[2]]
            colour_list = [self.colour_dict8[2], self.colour_dict8[5], self.colour_dict8[1], self.colour_dict8[2]]
            for l in range(len(lines_list)):
                lines_l = lines_list[l]
                for i in range(len(lines_l) - 1):
                    # if i not in [0,3]:
                    #     continue
                    p1 = lines_l[i]
                    p2 = lines_l[i + 1]
                    if self.point_cheak(p1, self.wPixel_real, self.hPixel_real) and self.point_cheak(p2,self.wPixel_real,self.hPixel_real):
                        # cv2.line(img_mark, tuple(p1), tuple(p2), (255, 255, 0), 2)
                        cv2.line(img_mark, tuple(p1), tuple(p2), colour_list[l], 2)

        # 画传送带和煤,全局渲染
        if True:
            # 画煤
            if len(lines1)>0 and len(lines2)>0:
                points_coals = np.concatenate((lines1, lines2[::-1]), axis=0)
                # 填充多边形
                cv2.fillPoly(self.count_like_label, [points_coals], (0, 0, 255))
                cv2.fillPoly(self.count_like_label_np, [points_coals], self.list_class.index('coal'))


            # 画传送带
            if len(lines2)>0 and len(lines3)>0:
                points_transmissionBelts = np.concatenate((lines2, lines3[::-1]), axis=0)
                # 填充多边形
                cv2.fillPoly(self.count_like_label, [points_transmissionBelts], (181, 119, 53))
                cv2.fillPoly(self.count_like_label_np, [points_transmissionBelts], self.list_class.index('transmissionBelt'))

        return img_mark

    def mark_4point(self,points_all, img_mark,):
        up_y = 50
        low_y = 100

        # self.hPixel_real ,w_pixel,_ = img_mark.shape
        points = []
        points_num_each_line = int(len(self.plate_Word)/2)
        plate_id = -int((len(points_all)/len(self.plate_Word)-1)/2)
        for point in points_all:
            points.append(point)
            if len(points)==len(self.plate_Word):
                right_point_num = self.points_cheak_right_num(points,self.wPixel_real, self.hPixel_real,0.5)
                # print('right_point_num',right_point_num)
                # if abs(plate_id)<5:
                if True:
                    if right_point_num == 0:
                        pass
                    # elif right_point_num == 2:
                    #     print('points',points)
                    elif right_point_num < 4:
                        pass
                    else:
                        # V_h = 100
                        points_Move_Up = [np.array([arr[0] if i < len(points)/2 else points[int(len(points)-i-1)][0], up_y if i < len(points)/2 else low_y], dtype=np.int16) for i, arr in enumerate(points)]
                        
                        points_Move_Up = points_Move_Up if self.flag_plate_UP else points

                        # 画中心点
                        if right_point_num == len(points) and abs(plate_id)<5:
                            x_draw_plate = int(sum([row[0] for row in points_Move_Up])/len(points))
                            y_draw_plate = int(sum([row[1] for row in points_Move_Up]) / len(points))
                            pixel_draw_row = 20
                            # print(str(plate_id),'w'+str(points_Move_Up[1][0]-points_Move_Up[0][0]),'h'+str(points_Move_Up[3][1] - points_Move_Up[0][1]))
                            if self.point_cheak((x_draw_plate,y_draw_plate), self.wPixel_real, self.hPixel_real):
                                # 在图片上添加文本标记序号
                                # 画架号
                                if True:
                                    text = str(plate_id+int(self.ID))
                                    font, font_scale, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                                    # 获取文本尺寸
                                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

                                    # 计算文本的起始点，使文本的中心与指定点对齐
                                    origin_x = x_draw_plate - text_width // 2
                                    origin_y = y_draw_plate + text_height // 2

                                    cv2.putText(img_mark,text, (origin_x,origin_y), font, font_scale, self.colour_dict8[1], font_thickness)
                                # 显示宽度
                                if False:
                                    cv2.putText(img_mark, 'w'+str(points_Move_Up[2][0]-points_Move_Up[0][0]), (x_draw_plate, y_draw_plate+pixel_draw_row), cv2.FONT_HERSHEY_SIMPLEX, 0.4,self.colour_dict8[0], 1)
                                    cv2.putText(img_mark, 'w'+str(points_Move_Up[3][0]-points_Move_Up[3][0]), (x_draw_plate, y_draw_plate+pixel_draw_row*2), cv2.FONT_HERSHEY_SIMPLEX, 0.4,self.colour_dict8[0], 1)
                                    cv2.putText(img_mark, 'h'+str(points_Move_Up[5][1] - points_Move_Up[0][1]), (x_draw_plate, y_draw_plate + pixel_draw_row*3),cv2.FONT_HERSHEY_SIMPLEX, 0.4,self.colour_dict8[0], 1)
                                    cv2.putText(img_mark, 'h'+str(points_Move_Up[3][1] - points_Move_Up[1][1]), (x_draw_plate, y_draw_plate + pixel_draw_row*4),cv2.FONT_HERSHEY_SIMPLEX, 0.4,self.colour_dict8[0], 1)
                        
                        # 标记点的序号
                        if False:
                            for i, point in enumerate(points_Move_Up):
                                if self.point_cheak(point, self.wPixel_real, self.hPixel_real) and i <4:
                                    x, y = point
                                    # 在图片上画点
                                    cv2.circle(img_mark, (x, y), 5, (0, 255, 0), -1)
                                    # 在图片上添加文本标记序号
                                    cv2.putText(img_mark, str(i), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                        # 按照顺序连接点 护帮板
                        if True :
                            if abs(plate_id)<5:
                                points_Move_Up.append(points_Move_Up[0])
                                for i in range(len(points_Move_Up) - 1):
                                    p1 = points_Move_Up[i]
                                    p2 = points_Move_Up[i + 1]
                                    if self.point_cheak(p1, self.wPixel_real, self.hPixel_real) and self.point_cheak(p2, self.wPixel_real, self.hPixel_real):
                                        # cv2.line(img_mark, tuple(p1), tuple(p2), (255, 255, 0), 2)
                                        cv2.line(img_mark, tuple(p1), tuple(p2), (255, 0, 0), 2)
                        # 画两条等高线
                        if True:
                            contour_line_1 = 0.6
                            contour_line_2 = 0.4

                            if self.point_cheak(points):
                                for i in range(len(points) - 1):
                                    ph11 = (int(points[0][0]+contour_line_1*(points[9][0]-points[0][0])),int(points[0][1]+contour_line_1*(points[9][1]-points[0][1])))
                                    ph12 = (int(points[4][0]+contour_line_1*(points[5][0]-points[4][0])),int(points[4][1]+contour_line_1*(points[5][1]-points[4][1])))
                                    if self.point_cheak(ph11, self.wPixel_real, self.hPixel_real) and self.point_cheak(ph12, self.wPixel_real, self.hPixel_real):
                                        # cv2.line(img_mark, tuple(p1), tuple(p2), (255, 255, 0), 2)
                                        cv2.line(img_mark, tuple(ph11), tuple(ph12), (0, 0, 255), 2)

                                    ph21 = (int(points[0][0]+contour_line_2*(points[9][0]-points[0][0])),int(points[0][1]+contour_line_2*(points[9][1]-points[0][1])))
                                    ph22 = (int(points[4][0]+contour_line_2*(points[5][0]-points[4][0])),int(points[4][1]+contour_line_2*(points[5][1]-points[4][1])))
                                    if self.point_cheak(ph21, self.wPixel_real, self.hPixel_real) and self.point_cheak(ph22, self.wPixel_real, self.hPixel_real):
                                        # cv2.line(img_mark, tuple(p1), tuple(p2), (255, 255, 0), 2)
                                        cv2.line(img_mark, tuple(ph21), tuple(ph22), (0, 255, 0), 2)
                        # 画护帮板
                        if True:
                            points_transmissionBelts = np.array([point for point in points if self.point_cheak(point, self.wPixel_real,self.hPixel_real, 0.5)]).astype(np.int32)
                            if len(points_transmissionBelts) > 2:
                                points_transmissionBelts = points_transmissionBelts[:, np.newaxis, :]
                                # cv2.drawContours(img_mark, [points_transmissionBelts], -1, (85, 255, 0), -1)
                                cv2.drawContours(self.count_like_label, [points_transmissionBelts], -1, (0, 255, 85), -1)
                                cv2.drawContours(self.count_like_label_np, [points_transmissionBelts], -1,self.list_class.index('plates'), -1)
                        
                        # 给杨哥的返回值(只转这个)
                        if True:
                            # points_ = [[p[0],100] for p in points]
                            points_ = points
                            top_left,bottom_right = None,None
                            for i in range(points_num_each_line):
                                if self.point_cheak(points_[i],self.wPixel_real,self.hPixel_real,0):
                                    top_left = [points[i][0],up_y]
                                    break
                            for j in range(points_num_each_line-1,0,-1):
                                if self.point_cheak(points_[j],self.wPixel_real,self.hPixel_real,0):
                                    bottom_right = [points[j][0],low_y]
                                    break
                            if (top_left is not None) and (bottom_right is not None) and j-i >= 1:
                                # print('i,j',i,j)
                                point_left_upper_right_lower = [top_left,bottom_right]
                                return_result = point_left_upper_right_lower,plate_id
                                # print('return_result',return_result,points[i][0],points[j][1])
                                # print('return_result',return_result)


                                # 画矩形 虚拟架号
                                cv2.rectangle(img_mark, top_left, bottom_right, (0, 255, 0), 2)

                                # 计算矩形中心点
                                center_x = (top_left[0] + bottom_right[0]) // 2
                                center_y = (top_left[1] + bottom_right[1]) // 2
                                text = str(plate_id)


                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 1
                                font_thickness = 2  

                                # 获取文本的宽度和高度
                                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

                                # 计算文本的左下角坐标（OpenCV的putText函数要求的）
                                text_x = center_x - text_width // 2
                                text_y = center_y + text_height // 2
                                cv2.putText(img_mark, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)



                points=[]
                plate_id+=1
        return img_mark


    def single_mark_lines(self,points_all, img_mark,ids):
        num_group= len(ids)
        points_num_each_group = int(len(points_all)/num_group)

        for id_index in range(num_group):
            points = points_all[id_index*points_num_each_group:(id_index+1)*points_num_each_group]
            right_point_num = self.points_cheak_right_num(points,self.wPixel_real, self.hPixel_real,0.5)
            # colour = self.colour_dict16[ids[id_index]%len(self.colour_dict16)]
            # colour = self.colour_dict16[16]
            colour = self.colour_dict8[2]

            if ids[id_index] != 4 or id_index % 2 == 0:
                continue
            if right_point_num == 0:
                pass
            if right_point_num < 4:
                pass
            else:
                # 按照顺序连接点
                if True :
                    for i in range(len(points)-1):
                        p1 = points[i]
                        p2 = points[i + 1]
                        if self.point_cheak(p1, self.wPixel_real, self.hPixel_real) and self.point_cheak(p2, self.wPixel_real, self.hPixel_real):
                            # cv2.line(img_mark, tuple(p1), tuple(p2), (255, 255, 0), 2)
                            cv2.line(img_mark, tuple(p1), tuple(p2), colour, 2)
                # 标记点的序号
                if False:
                    for i, point in enumerate(points):
                        x, y = point
                        # 在图片上画点
                        cv2.circle(img_mark, (x, y), 5, (0, 255, 0), -1)
                        # 在图片上添加文本标记序号
                        cv2.putText(img_mark, str(i), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
                # 写架号
                if True:
                    text = str(ids[id_index]+int(self.ID))
                    # text = str(ids[id_index])
                    font, font_scale, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    k = 0 if id_index % 2 == 0 else -1
                    x, y = points[k]
                    # cv2.putText(img_mark,text, (x, y), font, font_scale, colour, font_thickness)
                    cv2.putText(img_mark, text, (x+10, y), font, font_scale, colour, font_thickness)

            if True:  # 将同一个支架连接起来
                if id_index % 2 == 0:
                    p1 = points_all[id_index*points_num_each_group]
                    p2 = points_all[(id_index+1)*points_num_each_group]
                    if self.point_cheak(p1, self.wPixel_real, self.hPixel_real) and self.point_cheak(p2,
                                                                                                     self.wPixel_real,
                                                                                                     self.hPixel_real):
                        cv2.line(img_mark, tuple(p1), tuple(p2), colour, 2)

        return img_mark
    

    def rotation_matrix_x(self,angle_radians):
        """
        返回绕X轴旋转的旋转矩阵
        """
        c = math.cos(angle_radians)
        s = math.sin(angle_radians)
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])

    def rotation_matrix_y(self,angle_radians):
        """
        返回绕Y轴旋转的旋转矩阵
        """
        c = math.cos(angle_radians)
        s = math.sin(angle_radians)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])


    def rotation_matrix_z(self,angle_radians):
        """
        返回绕Z轴旋转的旋转矩阵
        """
        c = math.cos(angle_radians)
        s = math.sin(angle_radians)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])

    def RT_count_ori(self,Tilt,Pan):
        # 2.相机坐标系
        ## 2.1 平移
        tx = 0
        ty = 0
        tz = 0


        ## 2.2旋转
        θz = 0  # 相机水平角度，假设相机安装水平
        # Tilt += self._T_constant

        angle_x_radians = math.radians(Tilt)
        angle_y_radians = math.radians(Pan)
        angle_z_radians = math.radians(θz)

        Rx = self.rotation_matrix_x(angle_x_radians)
        Ry = self.rotation_matrix_y(angle_y_radians)
        Rz = self.rotation_matrix_z(angle_z_radians)

        if min(Tilt,Pan)<30 and False: #官方
            # R_3x3 = np.dot(Rz, np.dot(Ry, Rx)).astype(np.float32)  # 先执行Ry和Rx的乘法，然后将结果与Rz相乘
            R_3x3 = Rz @ Ry @ Rx
        else:
            R_3x3 = Rz @ Rx @ Ry
        if False:
            self.t = np.array([tx, ty, tz],dtype=np.float32)
            # 将 t 转换为 3x1 矩阵
            t = self.t.reshape((3, 1))
            # 2.3 刚体变换：平移+旋转,平移矩阵和旋转结果相乘的结果
            # 合并 R 和 t 形成 3x4 的外参矩阵
            Rt = np.hstack((R_3x3, t)).astype(np.float32)

            if self.flag_print_detal:
                print('T',Rt)
            return Rt
        else:
            return R_3x3


    def R_Rt(self,R_3x3):
        ## 2.1 平移
        tx = 0
        ty = 0
        tz = 0

        self.t = np.array([tx, ty, tz],dtype=np.float32)
        # 将 t 转换为 3x1 矩阵
        t = self.t.reshape((3, 1))
        # 2.3 刚体变换：平移+旋转,平移矩阵和旋转结果相乘的结果
        # 合并 R 和 t 形成 3x4 的外参矩阵
        Rt = np.hstack((R_3x3, t)).astype(np.float32)

        if self.flag_print_detal:
            print('T',Rt)
        return Rt




    def RT_count2(self,Tilt,Pan):
        # Pan = -Pan
        # 2.相机坐标系
        ## 2.1 平移
        tx = 0
        ty = 0
        tz = 0


        p = math.radians(Tilt)
        t = math.radians(Pan)

        # z = [cos(p) * cos(t), cos(p) * sin(t), -sin(p)]
        # x = [-sin(t), cos(t), 0]
        # y = [-sin(p) * cos(t), -sin(p) * sin(t), -cos(p)]
        # R = [x y z] 

        # 计算三角函数值
        cos_p = np.cos(p)
        sin_p = np.sin(p)
        cos_t = np.cos(t)
        sin_t = np.sin(t)

        # # 构建旋转矩阵
        # R_3x3 = np.array([
        #     [-sin_t,           -sin_p * cos_t,  cos_p * cos_t],
        #     [cos_t,            -sin_p * sin_t,  cos_p * sin_t],
        #     [0,                -cos_p,          -sin_p]
        # ])
        # 构建旋转矩阵
        # R_3x3 = np.array([
        #     [cos_t * cos_p,  sin_t,  -cos_t * sin_p],
        #     [-sin_t * cos_p, cos_t,  sin_t * sin_p],
        #     [sin_p,          0,      cos_p]
        # ])
        # 构建旋转矩阵
        # R_3x3 = np.array([
        #     [cos_t,           sin_t * sin_p,  -sin_t * cos_p],
        #     [0,               cos_p,           sin_p],
        #     [sin_t,          -cos_t * sin_p,   cos_t * cos_p]
        # ])
        R_3x3 = np.array([
            [cos_t,           sin_t * sin_p,  sin_t * cos_p],
            [0,               cos_p,           -sin_p],
            [-sin_t,          cos_t * sin_p,   cos_t * cos_p]
        ])
        self.t = np.array([tx, ty, tz],dtype=np.float32)
        # 将 t 转换为 3x1 矩阵
        tt = self.t.reshape((3, 1))
        # 2.3 刚体变换：平移+旋转,平移矩阵和旋转结果相乘的结果
        # 合并 R 和 t 形成 3x4 的外参矩阵
        Rt = np.hstack((R_3x3, tt)).astype(np.float32)

        if self.flag_print_detal:
            print('T',Rt)
        return Rt


    def RT_count3(self,Tilt,Pan):
        # Pan = -Pan
        # 2.相机坐标系
        ## 2.1 平移
        tx = 0
        ty = 0
        tz = 0
        p = math.radians(Tilt)
        t = math.radians(Pan)

        # 定义旋转向量
        rvec = np.array([p, t, 0.0])

        # 将旋转向量转换为旋转矩阵
        R_3x3, _ = cv2.Rodrigues(rvec)
        
        self.t = np.array([tx, ty, tz],dtype=np.float32)
        # 将 t 转换为 3x1 矩阵
        tt = self.t.reshape((3, 1))
        # 2.3 刚体变换：平移+旋转,平移矩阵和旋转结果相乘的结果
        # 合并 R 和 t 形成 3x4 的外参矩阵

        Rt = np.hstack((R_3x3, tt)).astype(np.float32)
        if self.flag_print_detal:
            print('T',Rt)
        return Rt




    def K_count(self):
        # 3 投影矩阵,内参矩阵
        cx = self.wPixel_referenc/2
        cy =self.hPixel_referenc/2
        K = [[self.fx, 0 ,cx],
            [0, self.fy, cy],
            [0, 0, 1]
             ]
        self.K = np.array(K).astype(np.float32)
        if self.flag_print_detal:
            print('self.K',self.K)
    # 需要转
    def count(self,Tilt,Pan,image1,np_predict=None):
        '''
        第一个参数 上下转（绕着x轴转）  单位：角度  该值为相对值，正对煤比为0，以下为正。
        第二个参数 左右转（绕着y轴转）  单位：角度  该值为相对值，正对煤比为0，以左为-，以右为+
        '''


        Pan = -Pan
        self.hPixel_real,self.wPixel_real,_ = image1.shape

        if True:
            if False:

                self.RT = self.RT_count_ori(Tilt + self._T_constant_camera, Pan)
                self.RT_support = self.RT_count_ori(Tilt + self._T_constant_support + self._T_constant_camera, Pan)
            else:
                self.RT = self.R_Rt(self.RT_count_ori(self._T_constant_camera, 0) @ self.RT_count_ori(Tilt,Pan))
                self.RT_support = self.R_Rt(self.RT_count_ori(self._T_constant_support + self._T_constant_camera,0) @ self.RT_count_ori(Tilt,Pan))
            # 这两种方法的计算结果大致一样（几乎一样）
        elif False:
            self.RT = self.RT_count2(Tilt+self._T_constant_camera,Pan)
            self.RT_support = self.RT_count2(Tilt+self._T_constant_support+self._T_constant_camera,Pan)
        else:
            self.RT = self.RT_count3(Tilt+self._T_constant_camera,Pan)
            self.RT_support = self.RT_count3(Tilt+self._T_constant_support+self._T_constant_camera,Pan)



        self.count_like_label = np.zeros([self.hPixel_real,self.wPixel_real,3]).astype(np.uint8)
        self.count_like_label_np = np.zeros([self.hPixel_real, self.wPixel_real]).astype(np.uint8)
        img_mark = image1
        if self.flag_test and np_predict is not None:
            mask = (np_predict == self.list_class.index('plates')) | (np_predict == self.list_class.index('transmissionBelt')) | (np_predict == self.list_class.index('coal'))
            # mask = (np_predict == self.list_class.index('plates')) | (np_predict == self.list_class.index('transmissionBelt'))
            # mask = (np_predict == self.list_class.index('transmissionBelt'))
            np_predict[~mask] = 0
            # self.np_predict = np_predict

            if self.np_predicts is None:
                self.np_predicts = np_predict[None, :, :]
            else:
                try:
                    self.np_predicts = np.vstack((self.np_predicts, np_predict[None, :, :]))
                except Exception as e:
                    # self.np_predicts None
                    self.flag_vstack = False
                    print('err996',e)

        if False:
            w_4d = self.all_points_Word[:len(self.plates_Word)]
            w_line = self.all_points_Word[len(self.plates_Word):len()]

            
            points_4d = self.World_pixel(w_4d,self.RT,self.K,)
            points_line = self.World_pixel(w_line,self.RT_support,self.K,)
        else:
            # 需要转
            points_4d = self.World_pixel(self.plates_Word,self.RT,self.K,)
            # 不需要转
            points_line = self.World_pixel(self.transmissionBelts_Word,self.RT_support,self.K,)
            # 需要转
            points_boundary = self.World_pixel(self.boundaries_Word,self.RT_support,self.K,)




        img_mark = self.mark_4point(points_4d, img_mark)
        img_mark = self.mark_lines(points_line, img_mark)
        ids = self.boundaries_id
        ids = [item for item in ids for _ in range(2)]
        # # 需要转240711
        img_mark = self.single_mark_lines(points_boundary,img_mark,ids)



        if self.count_like_label_nps is None:
            self.count_like_label_nps = self.count_like_label_np[None, :, :]
        else:
            try:
                self.count_like_label_nps = np.vstack((self.count_like_label_nps, self.count_like_label_np[None, :, :]))
            except Exception as e:
                print('err1027',e)
        
        return img_mark,self.count_like_label,self.count_like_label_np




    def count_loss(self,loss_name_path=None):
        self.print_jr('self.cam_num_plates,self.cam_num_transmissionBelts',self.cam_num_plates,self.cam_num_transmissionBelts)
        
        # self.np_predicts
        # self.count_like_label_nps
        if False:
            inputs = self.np_predicts
            targets = self.count_like_label_nps
            Class_IoU_new=None
            C = 8
            loss_jr = Loss_jr(inputs,targets,Class_IoU_new,C) # 此处的targets 就有边缘轮廓渐变的现象发生
            # findContours.Pass_ginseng(inputs,targets)
            loss_jr.jrFindContours()
            loss_mysef = loss_jr.loss_point_num()
            print('loss_mysef',loss_mysef)
        else:
            # if 'transmissionBelt' not in self.flag_compute_with: # 不要'transmissionBelt' 的区域算损失
            #     self.np_predicts[self.np_predicts == self.list_class.index('transmissionBelt')] = 0
            #     self.count_like_label_nps[self.count_like_label_nps == self.list_class.index('transmissionBelt')] = 0
            # if 'coal' not in self.flag_compute_with:# 不要coal 的区域算损失
            if True: #  不要coal 的区域算损失
                self.np_predicts[self.np_predicts == self.list_class.index('coal')] = 0
                self.count_like_label_nps[self.count_like_label_nps == self.list_class.index('coal')] = 0

            if 'all' not in self.flag_calculation_depends:
                calculation_npt_depends = set(self.list_class) - set(self.flag_calculation_depends) -set('background')
                for calculation_npt_depend in calculation_npt_depends: # calculation_npt_depend  的区域算损失
                    self.np_predicts[self.np_predicts == self.list_class.index(calculation_npt_depend)] = 0
                    self.count_like_label_nps[self.count_like_label_nps == self.list_class.index(calculation_npt_depend)] = 0
            else:
                pass

            # 使用 == 运算符比较两个数组

            equal_elements = self.np_predicts == self.count_like_label_nps
            # equal_elements = self.np_predicts == self.np_predicts
            # 使用 numpy.sum() 统计 True 的个数
            count_equal_elements = np.sum(equal_elements)
            Equal_proportion_elements = count_equal_elements/np.size(equal_elements)*100

            print(f"相等的元素个数比例: {Equal_proportion_elements} %")
            loss = 100-Equal_proportion_elements


            
            if loss_name_path is not None:
                assert len(loss_name_path) == equal_elements.shape[0]
                equal_elements_ = (equal_elements*255).astype(np.uint8)
                for i in range(len(loss_name_path)):
                    path_r = loss_name_path[i] + '_loss.jpg'
                    image = equal_elements_[i]
                    cv2.imwrite(path_r, image)
                print('len(loss_name_path)',len(loss_name_path))

        return loss



    def run_jr(self,Tilt,Pan,image1,mat_intri, coff_dis):
        # self.RT_count(Tilt, Pan)
        img_mark = image1
        if True:
            objectPoints = np.array(self.plates_Word) # 世界坐标系中的 3D 点
            rotationMatrix = self.R_3x3 # 描述：旋转矩阵，直接表示从世界坐标系到相机坐标系的旋转。
            tvec = self.t.astype(np.float32)# 平移向量，表示从世界坐标系到相机坐标系的平移。
            cameraMatrix = mat_intri # 相机内参矩阵，通常包含焦距 fx, fy（以像素为单位）和主点坐标 cx, cy（图像中心的坐标）。
            distCoeffs = coff_dis.astype(np.float32) # 畸变系数，一个 1x5、1x8 或 1x12 的数组。这些系数表示径向畸变（k1, k2, k3）和切向畸变（p1, p2, k4, k5, k6）。如果为 None，则不考虑畸变。
            points_pixel_repro, _ = cv2.projectPoints(objectPoints, rotationMatrix, tvec, cameraMatrix, distCoeffs)
            # imagePoints 现在包含了投影后的 2D 点
            imagePoints = points_pixel_repro.squeeze()  # 如果需要，将形状从(N, 1, 2)压缩到(N, 2)

            points = np.round(imagePoints).astype(np.int16)

        img_mark = self.mark(points, img_mark)
        # img_mark = self.mark(self.image_points_measure, img_mark)
        # cv2.imwrite('./3.jpg',img_mark)
        return img_mark


    def print_jr(self,*args):
        names = args[0].split(',')
        for i in range(len(names)):
            print(names[i],' : ',args[i+1])

    def main(self,root_path):
        # self.cam_num_transmissionBelts = 4 #12# 30#10#4 #100
        self.cam_num_transmissionBelts = 10  # 6#10
        self.cam_num_plates = 0#4
        self.init_jr()
        self.root_path = root_path
        result_path = root_path + '_result'
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        result_path_label = root_path + '_label'
        if not os.path.exists(result_path_label):
            os.mkdir(result_path_label)

        Vpath_aim = os.path.join(result_path, f'result_ID{self.ID}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')


        # video = cv2.VideoWriter(Vpath_aim, fourcc, 0.5, (self.wPixel_real, self.hPixel_real))
        video = None
        
        # camera_Principles = Camera_Principles()
        # 获取文件和目录列表
        files_and_dirs = os.listdir(root_path)

        # 使用sorted函数按照文件名排序
        sorted_files_and_dirs = sorted(files_and_dirs)
        

        # 输出排序后的列表
        

        for name  in sorted_files_and_dirs:
            if '_predict' in name :
                continue
            elif '.mp4' in name:
                continue
            # elif not '_T00_' in name:
            #     continue
            #
            # elif not 'ID68_T00_P182' in name:
            #     continue

            elif not str(self.ID) in name:
                continue

            print('name ', name)


            match = re.search(r'ID(\d+)_T(\d{2})_P(\d+).*.jpg', name)
            ID = int(match.group(1))
            Tilt = int(match.group(2))
            Pan = int(match.group(3))
            Pan = Pan - 180
            
            if str(self.ID) != str(ID):
                continue
            # elif abs(abs(Pan) - 90 )>10:
            #     continue
            # elif abs(Pan) > 25:
            #     continue
            # elif abs(Pan) < 50 or abs(Pan) > 130:
            #     continue

            # Tilt = 10
            # Pan = int(name.split('.')[0])
            # Pan = 180 - Pan

            image1 = cv2.imread(os.path.join(root_path, name))
            self.hPixel_real,self.wPixel_real,_ = image1.shape
            if video is None:
                video = cv2.VideoWriter(Vpath_aim, fourcc, 0.5, (self.wPixel_real, self.hPixel_real))
            
            if self.flag_test:
                try:
                    np_predict = np.load(os.path.join(root_path, name.split('.')[0] + '_predict.npy'))
                except:
                    np_predict = None
                    print(os.path.join(root_path, name.split('.')[0] + '_predict.npy'),'不存在')
                    
            
            # Tilt, Pan, = 15,30
            # Tilt, Pan, = -16.78,47.8

            # Tilt, Pan, = 30,-16
            # Tilt, Pan, = 45,-90
            # Tilt, Pan, = 45,0
            # Tilt, Pan, = 0,-97
            # Tilt, Pan, = 0,90
            # Tilt, Pan, = 90,90
            # Tilt, Pan, = 45,-45
            # Tilt, Pan, = 30,-90

            # 需要转
            # # 注意：这会导致像素值的简单相加，可能导致溢出，你可能需要对结果进行归一化
            # result = cv2.addWeighted(image1, 0.5, img_mark,0.5, 0)


            if self.flag_test:
                img_mark, img_mark_label, count_like_label_np = self.count(Tilt, Pan, image1,np_predict)
                result = img_mark
                # 保存结果图片
                path_r = os.path.join(result_path, name.split('.')[0] + '_T'+str(Tilt) +'_P' +str(Pan))
                cv2.imwrite(path_r + '.jpg', result)
                if np_predict is not None:

                    cv2.imwrite(path_r + '_img_mark_label.png', img_mark_label)
                    cv2.imwrite(path_r + '_count_like_label_np.png',count_like_label_np * 30)
                    cv2.imwrite(path_r + '_np_predict.png', np_predict * 30)

                    # 使用 == 运算符比较两个数组

                    # 检查形状是否匹配
                    if count_like_label_np.shape == np_predict.shape:
                        # equal_element = count_like_label_np == np_predict
                        pass
                    else:
                        # assert 1 == 2
                        print("Arrays do not have the same shape.")
                    equal_element = count_like_label_np == np_predict
                    cv2.imwrite(os.path.join(path_r + '_np_equal_element.png'), equal_element * 255)

                video.write(result)
            else:
                self.count(Tilt, Pan, image1)
        if self.flag_test :
            if video is not None:
                video.release()  # 释放
            try:
                loss = self.count_loss()
                print('loss', loss)
                return loss
            except:
                print('err loss')

            

    def calculate_error(self,):
        result_path = self.root_path + '_result'
        loss_name_path = []
        for name in os.listdir(self.root_path):
            if '_predict' in name:
                continue
            elif '.mp4' in name:
                continue    
            elif not str(self.ID) in name:
                continue
            if self.flag_print_detal:
                print('name ', name)
            # match = re.search(r'ID(\d{2})_T(\d{2})_P(\d+).*.jpg', name)
            match = re.search(r'ID(\d+)_T(\d{2})_P(\d+).*.jpg', name)
            if match is None:
                continue

            ID = int(match.group(1))
            Tilt = int(match.group(2))
            Pan = int(match.group(3))
            # Tilt = 10
            # Pan = int(name.split('.')[0])
            if str(self.ID) != str(ID):
                continue
            Pan = Pan - 180
            # if abs(Pan) > 25 and 'plates' in self.flag_calculation_depends and len(self.flag_calculation_depends)==1:
            #     continue


            # if abs(Pan) < 50 or abs(Pan) > 130:
            #     continue
            # elif abs(Pan) > 130:
            #     continue
            np_predict = np.load(os.path.join(self.root_path, name.split('.')[0] + '_predict.npy'))
            image1 = cv2.imread(os.path.join(self.root_path, name))
            self.count(Tilt, Pan, image1,np_predict)
            path_r = os.path.join(result_path, name.split('.')[0] + '_T'+str(Tilt) +'_P' +str(Pan))
            
            loss_name_path.append(path_r)
        # 
        if self.flag_vstack:
            loss = self.count_loss(loss_name_path)
        else:
            loss = None
        self.error_function_N0+=1
        print('\nID:',self.ID,' loss:', loss, self.error_function_N0,'平均花费时间：',(time.time()-self.time_begin)/self.error_function_N0)
        return loss


    # 自定义函数，计算误差
    def error_function(self,params):
        # params是一个包含5个未知参数的数组
        # 参数变少，时间减少是指数级的
        if 'all' in self.flag_calculation_depends:
            self.d_level, self.camera_h_word, self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support = params
        elif 'plates' in self.flag_calculation_depends and len(self.flag_calculation_depends)==1:
            self.d_level,self._T_constant_camera = params
        elif 'transmissionBelt' in self.flag_calculation_depends and len(self.flag_calculation_depends) == 1:
            self.camera_h_word, self.transmissionBelt_Coal_wall1,self._T_constant_support = params
        else:
            self.camera_h_word, self.transmissionBelt_Coal_wall1,self._T_constant_support = params
        self.init_jr()
        # self.print_jr('self.flag_calculation_depends,self.d_level, self.camera_h_word, self.transmissionBelt_Coal_wall1,self._T_constant_camera,
        # self._T_constant_support',self.flag_calculation_depends,self.d_level, self.camera_h_word, self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support,)

        t_temp = time.time()

        # 这里替换为你的误差计算逻辑
        # 假设 calculate_error 是你计算误差的函数
        error = self.calculate_error()
        print('单次花费时间：',time.time() - t_temp)
        return error
    


    def minimize_count(self):
        self.cam_num_transmissionBelts = 10#6#10
        self.cam_num_plates = 8#20
        self.init_jr()
        # 初始猜测值，可以是任意合法的初始参数值
        # initial_guess = np.zeros(5)  # 例如，全部设为0

        # initial_guess = np.array([self.d_level,self.camera_h_word,self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support]).astype(np.float16)
        # initial_guess = np.array([self.d_level,self.camera_h_word,self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support]).astype(np.int32)
        print('self.flag_calculation_depends',self.flag_calculation_depends)


        if 'all' in self.flag_calculation_depends:
            params = self.d_level, self.camera_h_word, self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support
            list_temp = [[1e3, 10e3],[1e3, 5e3],[1e3, 5e3],[-20,20],[-20,20]]    
        elif 'plates' in self.flag_calculation_depends and len(self.flag_calculation_depends)==1:
            params = self.d_level,self._T_constant_camera
            list_temp = [[1e3, 10e3],[-20,20]]
        elif 'transmissionBelt' in self.flag_calculation_depends and len(self.flag_calculation_depends) == 1:
            params = self.camera_h_word, self.transmissionBelt_Coal_wall1,self._T_constant_support
            list_temp = [[1e3, 5e3], [1e3, 5e3], [-20, 20]]
        else:
            params = self.camera_h_word, self.transmissionBelt_Coal_wall1,self._T_constant_support
            list_temp = [[1e3, 5e3],[1e3, 5e3],[-20,20]]

        initial_guess = params

        # 约束
        # 边界约束（可选）
        # list_temp = [[1e3, 10e3],[1e3, 5e3],[1e3, 5e3],[-30,30],[-30,30]]        

        bounds_mins = [l[0] for l in list_temp]
        bounds_maxs = [l[1] for l in list_temp]
        bounds = Bounds(bounds_mins,bounds_maxs)
        if False:
            # 约束条件
            def constraint1(x):
                return x[0]
            def constraint2(x):
                return x[1]
            # 定义约束字典
            con1 = {'type': 'ineq', 'fun': constraint1}
            con2 = {'type': 'ineq', 'fun': constraint2}
            cons = [con1, con2]

        # 设置优化参数
        # options = {'maxiter': 50}  # 最大迭代次数,100
        # # 优化选项
        options = {
            'maxiter': 100,#50,     # 最大迭代次数 100
            'maxfun': 50,      # 最大函数评估次数 150
            'ftol': 1e-6,       # 函数值容差
            'xtol': 1e-6,       # 参数值容差
            'gtol': 1e-6,        # 梯度容差
            'disp': True         # 打印优化过程
        }

        # 使用不同的方法进行优化
        methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr']
        # L-BFGS-B 不收敛
        # 使用Nelder-Mead法进行优化
        # result = minimize(self.error_function, initial_guess, method='nelder-mead',bounds=bounds, options=options)#constraints=cons
        result = minimize(self.error_function, initial_guess, method='COBYLA',bounds=bounds, options=options)#constraints=cons

        # 输出最优解
        print("最优参数:", result.x)
        print("最小误差:", result.fun)

        guess = np.array([self.d_level, self.camera_h_word, self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support]).astype(np.float16)
        np.save(self.name_initial_guess,guess)  # 保存文件

    
    def resolution_conversion(self,points):
        points[:,0]*=self.wPixel_real/self.wPixel_referenc
        points[:,1]*=self.hPixel_real/self.hPixel_referenc
        return points



    # 不转
    if False:
        def minimize_count_plates(self):
            # 初始猜测值，可以是任意合法的初始参数值
            # initial_guess = np.zeros(5)  # 例如，全部设为0

            # initial_guess = np.array([self.d_level,self._T_constant_camera]).astype(np.float16)
            initial_guess = np.array([self.d_level,self._T_constant_camera]).astype(np.int32)

            # 约束ddd
            # 边界约束（可选）
            list_temp = [[1e3, 10e3],[-30,30]]        

            bounds_mins = [l[0] for l in list_temp]
            bounds_maxs = [l[1] for l in list_temp]
            bounds = Bounds(bounds_mins,bounds_maxs)
            # 设置优化参数
            options = {'maxfun': 50}  # 最大迭代次数,100

            # 使用不同的方法进行优化
            methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr']


            # 使用Nelder-Mead法进行优化
            result = minimize(self.error_function_plates, initial_guess, method='nelder-mead',bounds=bounds, options=options)#constraints=cons
            # result = minimize(self.error_function_plates, initial_guess, method='L-BFGS-B',bounds=bounds, options=options)#constraints=cons

            # 输出最优解
            print("最优参数:", result.x)
            print("最小误差:", result.fun)


        def error_function_plates(self,params):
            # params是一个包含5个未知参数的数组
            self.d_level,self._T_constant_camera = params
            self.print_jr('self.d_level, self.camera_h_word, self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support',
                          self.d_level, self.camera_h_word, self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support)

            t_temp = time.time()

            # 这里替换为你的误差计算逻辑
            # 假设 calculate_error 是你计算误差的函数

            error = self.calculate_error()
            print('单次花费时间：',time.time() - t_temp)
            # np.save(self.name_initial_guess,params)  # 保存文件
            return error
        
        def point_Calculate_angle(self,point):
            x,y,z = point
            r = math.sqrt(x**2 + y**2 + z**2)
            a_LR = math.degrees(math.atan(x/z))
            a_UP = math.degrees(math.atan(y/z))

            return a_UP,a_LR
        def quadrilateral_centroid(self,vertices):
            x_coords = [vertex[0] for vertex in vertices]
            y_coords = [vertex[1] for vertex in vertices]
            z_coords = [vertex[2] for vertex in vertices]
            centroid_x = sum(x_coords) / len(vertices)
            centroid_y = sum(y_coords) / len(vertices)
            centroid_z = sum(z_coords) / len(vertices)
            return centroid_x, centroid_y,centroid_z
        def Calculate_degrees_see(self,points):
            center = self.quadrilateral_centroid(points)
            # for point in points:
            #     a_UP,a_LR = self.point_Calculate_angle(point)
            a_UP,a_LR = self.point_Calculate_angle(center)
            RT = self.RT_count(a_UP,a_LR)
            
            # points_region = self.World_pixel(self.boundaries_Word,RT,self.K,)
            points_region = self.World_pixel(points,RT,self.K,)
            right_point_num = self.points_cheak_right_num(points_region,self.wPixel_referenc,self.hPixel_referenc,0.5)
            if right_point_num == len(points):
                print('该相机可以看全',right_point_num,len(points))
            else:
                print('该相机看不全',right_point_num,len(points))
            print('a_UP,a_LR',a_UP,a_LR)
            self.print_jr('points,points_region',points,points_region)
            

def long_time_task(ID,root_path):
    flag_minimize_count = True
    while flag_minimize_count:
    # if True:
    # for ID in IDs:
        # 需要转
        camera_Principles = Camera_Principles(ID,root_path)
        # 需要转
        camera_Principles.main(root_path)
        del camera_Principles

        flag_minimize_count = False
        if platform.system() == 'Linux' or True:
            flag_minimize_count = True
            if False:
                # # 不需要转
                # camera_Principles.minimize_count()
                # camera_Principles.minimize_count_plates()
                camera_Principles = Camera_Principles(ID,root_path)
                camera_Principles.flag_calculation_depends = ['plates']
                camera_Principles.minimize_count()
                del camera_Principles
                
            if False:
                # 不需要转
                camera_Principles = Camera_Principles(ID,root_path)
                camera_Principles.flag_calculation_depends = ['plates','transmissionBelt',]
                camera_Principles.minimize_count()
                del camera_Principles
            if False:
                # 不需要转
                camera_Principles = Camera_Principles(ID,root_path)
                camera_Principles.flag_calculation_depends = ['all']
                camera_Principles.minimize_count()
                del camera_Principles

            if True:
                # 不需要转
                camera_Principles = Camera_Principles(ID,root_path)
                camera_Principles.flag_calculation_depends = ['transmissionBelt',]
                camera_Principles.minimize_count()
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
        root_path = r'D:\dataset\Virtual_plate_number\all'
    # IDs = [44,68]
    IDs = [68,44]
    # IDs = [68,44,103,118,999]
    # IDs = [103,118,999]
    # IDs = [44]
    # points = [[0,0.0,10000],[500,500,10000],[100,100,10000]]
    # points = [[0,0.0,10000],[0,0,10000]]
    # points = np.array(points)
    if True:
        for ID in IDs:
            # # 创建子进程
            if True and platform.system() == 'Linux':
                p = Process(target=long_time_task, args=(ID,root_path))  # target进程执行的任务, args传参数（元祖）
                p.start()   # 启动进程
            else:
                long_time_task(ID,root_path)

    # camera_Principles = Camera_Principles()
    # camera_Principles.Calculate_degrees_see(points)

'''

****z****
想法：
能不能根据特定点位求（类似于相机标定的高效和高精度），比如
通过旋转90度，无穷远处的位置，确定相机安装的距离，和偏转的角度
借助无穷远处的圆或者圆心，校准参数
********


量中心距离的时候，多量几架，除以架数，可以减少误差

移架，会使得煤壁距离不准



****old***
自己写相机的畸变部分，k1,k2,k3,k4 (不用了)
****old***


'''

