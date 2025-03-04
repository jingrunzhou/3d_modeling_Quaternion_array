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
from quaternion_array_ok import QuaternionCamera,ViewTransform,transform_point_to_screen
from Euler_angle import Euler

class Camera_Principles():
    def __init__(self, ID=None, root_path=None):
        # 设置
        self.flag_Move_first = True
        self.list_class = ['background', 'plates', 'transmissionBelt', 'track', 'cableTrough', 'groovedSide', 'coal','roller']
        self.np_predicts = None
        self.count_like_label_nps = None
        self.ID = str(ID)
        self.root_path = root_path
        self.name_initial_guess = f"./guess/initial_guess_ID{self.ID}.npy"
        # self.name_initial_guess = f"./guess/oldf_241111/initial_guess_ID{self.ID}.npy"
        self.flag_calculation_depends = []
        self.time_begin = time.time()
        self.flag_plate_UP = False
        self.flag_test = True #False  # 调试模式开关  如果不想画图，不想测试，该值为False,
        self.wPixel_referenc = 704
        self.hPixel_referenc = 576
        self.flag_vstack = True
        if False:  # 51
            self.fx, self.fy = [468.61, 680.34]  # [593.74,658.35]#[542.9,629.46]
            self.dist = np.array([-4.24103449e-01, 2.11013612e-01, -1.43886286e-03, -3.49882761e-05, -5.33531165e-02], dtype=np.float32)
        else:  # all
            self.fx, self.fy = [468.64, 680.222]
            self.dist = np.array([-4.23709796e-01, 2.23283964e-01, -6.57576014e-04, -4.92667899e-05, -6.80752617e-02],dtype=np.float32)
        self.k1, self.k2, self.p1, self.p2, self.k3 = self.dist
        # self.Camera_internal_reference = self.fx, self.fy,self.k1, self.k2, self.p1, self.p2, self.k3
        # 不需要转
        if os.path.exists(self.name_initial_guess) and self.flag_test and True:
            self.d_level, self.installHeight, self.transmissionBelt_Coal_wall1, self._T_constant_camera, self._T_constant_support = np.load(
                self.name_initial_guess, allow_pickle=True)
            print(f'#{self.ID}加载npy参数成功')
        # 需要转
        else:
            # raise 'os.path.exists(self.name_initial_guess) and self.flag_test'
            # self.d_level, self.installHeight, self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support  = 3390,2500,1872,12,-2.6
            # self.d_level, self.installHeight, self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support  = 3390,2500,1872,0,0
            # self.d_level, self.installHeight, self.transmissionBelt_Coal_wall1, self._T_constant_camera, self._T_constant_support = 3390, 2500, 1872, 0, 0
            # self.d_level, self.installHeight, self.transmissionBelt_Coal_wall1, self._T_constant_camera, self._T_constant_support = 3800, 4000, 1400, 0, 0
            # self.d_level, self.installHeight, self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support  = 3390,2500,1872,0,0
            self.d_level, self.installHeight, self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support  = 3000,2500,1872,0,0
            print(f'#{self.ID}加载初始参数')
            # if platform.system() == 'Linux':
            #     raise '44'
        if False and platform.system() == 'Windows':  # temp ＩＤ４４　 画图transmissionBelt_Coal
            if self.ID == '44':
                # self._T_constant_support = -10
                # self._T_constant_camera -= 8
                self._T_constant_support = -self._T_constant_camera - 8
                # self.d_level += 1000
                # self.transmissionBelt_Coal_wall1 +=
                # self.installHeight -= 800
                self.transmissionBelt_Coal_wall1 -= 100
        self.error_function_N0 = 0
        self.flag_print_detal = False
        # 设备型号：DS-2DC2D40IW-DE3
        # 1、一些参数
        # 1.1、无误差值
        self.w_plate_word = 1400  # self.w_plate_word 护帮板实际宽度，
        self.h_plate_word = 1340  # self.h_plate_wor 护帮板实际高度
        self.Unwanted_h_plate_word = -300#300
        self.w_transmissionBelt_word = 800  # 1000 #800#1500#800#1500 #17000 #800
        self.f = 2.8
        self.HFOV_half = 97 / 2  # —— 水平视场角（Horizontal Field of View）的一半
        self.transmissionBelt_Coal_wall0 = 0
        self.y_higher_plate = 0  # -100#-300
        # self.transmissionBelt_Coal_wall3 = 4.5e3 #5.5e3
        self.cam_num_plates = 0  # 8#10#13 # 10#15#20 #20
        self.cam_num_transmissionBelts = 0  # 10 #10#16#10#13 # 10#15#20 #20
        # boundaries_id = [0,1,2,3,4,6,7,8,9,12,16,25,50]
        boundaries_id = [0, 1, 2, 3, 4, 6, 7, 8, 9, 12]
        boundaries_id = boundaries_id + [-x for x in boundaries_id]
        self.boundaries_id = sorted(set(boundaries_id))
        self.Installation_offset = 0  # -800 # 相机安装的偏移位置，从支架视角看，以左为负，右为正，安装到支架中间，该值为0。单位：mm。
        # self.wPixel_real = self.wPixel_referenc  # 2000 #704 # 实际像素值
        # self.hPixel_real = self.hPixel_referenc  # 2000 # 576 # 实际像素值
        # self.wPixel_real = 128000  # 2000 #704 # 实际像素值
        # self.hPixel_real = 128000 # 2000 # 576 # 实际像素值
        '''
        installHeight="-1" coalMiningAreaWidth="-1" bigFootWidth="-1"   三个参数，摄像头安装高度，采煤区域宽度，大脚区域宽度
        '''
        self.bigFootWidth = 1828
        # self.transmissionBelt_Coal_wall2 = self.transmissionBelt_Coal_wall1 + self.w_transmissionBelt_word
        self.coalMiningAreaWidth = self.transmissionBelt_Coal_wall1 + self.w_transmissionBelt_word  # 需要给定
        self.transmissionBelt_Coal_wall2 = self.coalMiningAreaWidth
        self.transmissionBelt_Coal_wall3 = self.coalMiningAreaWidth + self.bigFootWidth
        self.evler = Euler(self.wPixel_referenc,self.hPixel_referenc,self.fx,self.fy,self.dist,self.flag_print_detal)
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
            colour_wh = int(len(self.colour_dict) ** (1 / 2)) + 2
            colour_img = np.zeros([colour_wh * colour_interval, colour_wh * colour_interval, 3])
            for k in self.colour_dict.keys():
                y = (k // (colour_wh - 1) + 1) * colour_interval
                x = (k % (colour_wh - 1) + 1) * colour_interval
                # 在图片上画点
                cv2.circle(colour_img, (x, y), 20, self.colour_dict[k], -1)
                # 在图片上添加文本标记序号
                cv2.putText(colour_img, str(k), (x - 30, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colour_dict[k], 1)
            cv2.imwrite('./5.jpg', colour_img)
    # 需要转
    def init_jr(self):
        self.np_predicts = None
        self.count_like_label_nps = None
        # self.W_half_pixel = self.wPixel_real / 2
        # 1.2、有误差值的观测者
        # 9203,68# T00 w191h226
        self.w_plate_pixel = 194  # 191#200 #205#201 #165
        self.h_plate_pixel = 227  # 226#250 #272#256 #250
        self.interval_plate_pixel = 46  # 43#41# (41+31)/2#43
        # 1.3、有误差值的推测值
        # self.transmissionBelt_Coal_wall1 = 1500 #1500#1500#1500
        # 1.3、有误差值的计算值
        if False:
            self.interval_plate_word = (self.w_plate_word / self.w_plate_pixel) * self.interval_plate_pixel
        else:
            self.interval_plate_word = 332
        # 大了往里,小了往外
        # 2.0 最少需要求解的参数
        '''
        self._T_constant_camera = 6#7#12#10 #5 #10 # -5 #相机向下偏转的角度误差 # 远处往下跑，要调大
        self.d_level 
        self.installHeight
        self.transmissionBelt_Coal_wall1
        self.transmissionBelt_Coal_wall2
        '''
        # 单位mm,毫米
        # 1.世界坐标系
        self.plate_Word = [
            [-self.w_plate_word / 2, self.Unwanted_h_plate_word, 0],
            [-self.w_plate_word / 4, self.Unwanted_h_plate_word, 0],
            [0, self.Unwanted_h_plate_word, 0],
            [self.w_plate_word / 4, self.Unwanted_h_plate_word, 0],
            [self.w_plate_word / 2, self.Unwanted_h_plate_word, 0],
            [self.w_plate_word / 2, self.Unwanted_h_plate_word+self.h_plate_word, 0],
            [self.w_plate_word / 4, self.Unwanted_h_plate_word+self.h_plate_word, 0],
            [0, self.Unwanted_h_plate_word+self.h_plate_word, 0],
            [-self.w_plate_word / 4, self.Unwanted_h_plate_word+self.h_plate_word, 0],
            [-self.w_plate_word / 2, self.Unwanted_h_plate_word+self.h_plate_word, 0],
        ]
        # self.installHeight -=1000
        # self.transmissionBelt_Coal_wall1 -=1000
        hh_plate = 200
        h_dm = 200
        Ground_error = 0  # 800
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
            # # 墙线
            # [-self.w_plate_word / 2, self.installHeight, -self.transmissionBelt_Coal_wall0],
            # [-self.w_plate_word / 4, self.installHeight, -self.transmissionBelt_Coal_wall0],
            # [0, self.installHeight, -self.transmissionBelt_Coal_wall0],
            # [self.w_plate_word / 4, self.installHeight, -self.transmissionBelt_Coal_wall0],
            # [self.w_plate_word / 2, self.installHeight, -self.transmissionBelt_Coal_wall0],
            # 传送带线1
            [-self.w_plate_word / 2, self.installHeight, -self.transmissionBelt_Coal_wall1],
            [-self.w_plate_word / 4, self.installHeight, -self.transmissionBelt_Coal_wall1],
            [0, self.installHeight, -self.transmissionBelt_Coal_wall1],
            [self.w_plate_word / 4, self.installHeight, -self.transmissionBelt_Coal_wall1],
            [self.w_plate_word / 2, self.installHeight, -self.transmissionBelt_Coal_wall1],
            # 传送带线2
            [-self.w_plate_word / 2, self.installHeight, -self.transmissionBelt_Coal_wall2],
            [-self.w_plate_word / 4, self.installHeight, -self.transmissionBelt_Coal_wall2],
            [0, self.installHeight, -self.transmissionBelt_Coal_wall2],
            [self.w_plate_word / 4, self.installHeight, -self.transmissionBelt_Coal_wall2],
            [self.w_plate_word / 2, self.installHeight, -self.transmissionBelt_Coal_wall2],
            # 大脚趾线
            [-self.w_plate_word / 2, self.installHeight, -self.transmissionBelt_Coal_wall3],
            [-self.w_plate_word / 4, self.installHeight, -self.transmissionBelt_Coal_wall3],
            [0, self.installHeight, -self.transmissionBelt_Coal_wall3],
            [self.w_plate_word / 4, self.installHeight, -self.transmissionBelt_Coal_wall3],
            [self.w_plate_word / 2, self.installHeight, -self.transmissionBelt_Coal_wall3],
        ]
        self.boundary_Word = [
            # # 竖直在煤壁上的一条线
            # [0, 0, 0],
            # [0, self.Unwanted_h_plate_word, 0],
            # [0, self.h_plate_word, 0],
            # [0, (self.h_plate_word + self.installHeight)/2, 0],
            # [0, self.installHeight, 0],
            # 地面上的线
            # [0, self.installHeight, Ground_error],
            # [0, self.installHeight, -self.transmissionBelt_Coal_wall0],
            # [0, self.installHeight, -self.transmissionBelt_Coal_wall1],
            [0, self.installHeight, -self.transmissionBelt_Coal_wall2],
            [0, self.installHeight, -self.d_level],
            [0, self.installHeight, -self.transmissionBelt_Coal_wall3],
            # [0, self.installHeight, -5e3],
            # [0, self.installHeight, -6e3],
            # [0, self.installHeight, -7e3],
        ]
        self.plate_fold_r = 0.2 #0.3#0.2 #0.4#0.5
        # self.plate_fold_Word_old = [
        #     [-self.w_plate_word / 2, self.Unwanted_h_plate_word + self.plate_fold_r*(self.h_plate_word-self.Unwanted_h_plate_word), 0],
        #     [-self.w_plate_word / 4, self.Unwanted_h_plate_word+ self.plate_fold_r*(self.h_plate_word-self.Unwanted_h_plate_word), 0],
        #     [0, self.Unwanted_h_plate_word+ self.plate_fold_r*(self.h_plate_word-self.Unwanted_h_plate_word), 0],
        #     [self.w_plate_word / 4, self.Unwanted_h_plate_word+ self.plate_fold_r*(self.h_plate_word-self.Unwanted_h_plate_word), 0],
        #     [self.w_plate_word / 2, self.Unwanted_h_plate_word+ self.plate_fold_r*(self.h_plate_word-self.Unwanted_h_plate_word), 0],
        # ]
        self.plate_fold_Word_demo = [self.w_plate_word / 2, self.Unwanted_h_plate_word + self.plate_fold_r*self.h_plate_word, 0]
        split_num = 50
        self.plate_fold_Word = []
        for s in range(-split_num,split_num+1,1):
            self.plate_fold_Word.append([self.plate_fold_Word_demo[0]*s/split_num,self.plate_fold_Word_demo[1],self.plate_fold_Word_demo[2]])
        self.plates_Word = []
        # self.plates_Word_calibration = []
        self.transmissionBelts_Word = []
        self.boundaries_Word = []
        self.plates_fold_Word = []
        self.plate_center_distance = self.w_plate_word + self.interval_plate_word  #
        Z_Coal_wall_word = self.d_level if self.flag_Move_first else 0
        for i in range(-self.cam_num_plates, self.cam_num_plates + 1, 1):
        # for i in range(-self.cam_num_plates, -self.cam_num_plates+1, 1):
            for wi in self.plate_Word:
                self.plates_Word.append(
                    [wi[0] + (self.w_plate_word + self.interval_plate_word) * i, wi[1] + self.y_higher_plate,
                     wi[2] + Z_Coal_wall_word])
        for i in range(-self.cam_num_plates, self.cam_num_plates + 1, 1):
            for wi in self.plate_fold_Word:
                self.plates_fold_Word.append(
                    [wi[0] + (self.w_plate_word + self.interval_plate_word) * i, wi[1] + self.y_higher_plate,
                     wi[2] + Z_Coal_wall_word])
        for i in range(-self.cam_num_transmissionBelts, self.cam_num_transmissionBelts + 1, 1):
            for wi in self.transmissionBelt_Word:
                self.transmissionBelts_Word.append(
                    [wi[0] + (self.w_plate_word + self.interval_plate_word) * i, wi[1] + self.y_higher_plate,
                     wi[2] + Z_Coal_wall_word])
        for i in self.boundaries_id:
            for wi in self.boundary_Word:
                # self.boundaries_Word.append([wi[0] + (self.w_plate_word + self.interval_plate_word) * i,wi[1] + self.y_higher_plate, wi[2]+Z_Coal_wall_word])
                self.boundaries_Word.append(
                    [wi[0] - self.w_plate_word / 2 + (self.w_plate_word + self.interval_plate_word) * i,
                     wi[1] + self.y_higher_plate, wi[2] + Z_Coal_wall_word])
            for wi in self.boundary_Word:
                self.boundaries_Word.append(
                    [wi[0] + self.w_plate_word / 2 + (self.w_plate_word + self.interval_plate_word) * i,
                     wi[1] + self.y_higher_plate, wi[2] + Z_Coal_wall_word])
        # self.all_points_Word = self.plates_Word + self.transmissionBelts_Word + self.boundaries_Word
        self.plates_Word = [[-self.Installation_offset + p[0], 0 + p[1], 0 + p[2]] for p in self.plates_Word]
        self.plates_fold_Word = [[-self.Installation_offset + p[0], 0 + p[1], 0 + p[2]] for p in self.plates_fold_Word]
        self.transmissionBelts_Word = [[-self.Installation_offset + p[0], 0 + p[1], 0 + p[2]] for p in
                                       self.transmissionBelts_Word]
        self.boundaries_Word = [[-self.Installation_offset + p[0], 0 + p[1], 0 + p[2]] for p in self.boundaries_Word]
        flag_quaternion_array = True
        if flag_quaternion_array:#quaternion_array 不一样
            words = [self.plates_Word,self.plates_fold_Word,self.transmissionBelts_Word,self.boundaries_Word]
            for i in range(len(words)):
                words[i] = [[p[0], -p[1], -p[2]] for p in words[i]]
            self.plates_Word,self.plates_fold_Word,self.transmissionBelts_Word,self.boundaries_Word = words
        self.evler.K_count()
        self.print_jr(
            'self.flag_calculation_depends,self.d_level, self.installHeight, self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support',
            self.flag_calculation_depends, self.d_level, self.installHeight, self.transmissionBelt_Coal_wall1,
            self._T_constant_camera, self._T_constant_support, )
    def point_cheak(self, point, w_pixel=None, h_pixel=None, k=0.5):
        if np.isnan(point).any():
            return False
        elif np.any(point == np.int16(-32768)):
            return False
        if w_pixel is None:
            return True
        else:
            if -w_pixel * k < point[0] < w_pixel * (1 + k) and -h_pixel * k < point[1] < h_pixel * (1 + k):
                return True
            else:
                return False
    def points_cheak_right_num(self, points, w_pixel=None, h_pixel=None, k=0.5):
        right_num = 0
        for point in points:
            if self.point_cheak(point, w_pixel, h_pixel, k):
                right_num += 1
        return right_num
    def mark_lines(self, points_all, img_mark, num_line_each_group=4):
        # points_num_each_line = int(len(self.transmissionBelt_Word)/num_line_each_group)
        points_num_each_line = 5
        num_line_each_group = int(len(self.transmissionBelt_Word) / points_num_each_line)
        points_num_each_group = num_line_each_group * points_num_each_line
        num_group = int(len(points_all) / points_num_each_group)
        # 按照顺序连接点
        lines1 = np.array([points_all[n] for n in range(len(points_all)) if
                           n % points_num_each_group < points_num_each_line and self.point_cheak(points_all[n],
                                                                                                 self.wPixel_real,
                                                                                                 self.hPixel_real,
                                                                                                 0.5)]).astype(np.int32)
        lines2 = np.array([points_all[n] for n in range(len(points_all)) if
                           points_num_each_line <= n % points_num_each_group < points_num_each_line * 2 and self.point_cheak(
                               points_all[n], self.wPixel_real, self.hPixel_real, 0.5)]).astype(np.int32)
        lines3 = np.array([points_all[n] for n in range(len(points_all)) if
                           points_num_each_line * 2 <= n % points_num_each_group < points_num_each_line * 3 and self.point_cheak(
                               points_all[n], self.wPixel_real, self.hPixel_real, 0.5)]).astype(np.int32)

        lines4 = np.array([points_all[n] for n in range(len(points_all)) if
                           points_num_each_line * 3 <= n % points_num_each_group and self.point_cheak(points_all[n],self.wPixel_real,
                                                                                                      self.hPixel_real,
                                                                                                      0.5)]).astype(np.int32)
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
                    list_p = [p0, p1, p2, p3, p4]
                    # list_p = [p0,p1,p2]
                    if True:
                        for pi in range(len(list_p) - 1):
                            if self.point_cheak(list_p[pi], self.wPixel_real, self.hPixel_real) and self.point_cheak(
                                    list_p[pi + 1], self.wPixel_real, self.hPixel_real):
                                cv2.line(img_mark, tuple(list_p[pi]), tuple(list_p[pi + 1]), self.colour_dict8[2], 2)
                        if False:
                            if self.point_cheak(p1, self.wPixel_real, self.hPixel_real) and self.point_cheak(p2, self.wPixel_real,
                                                                                                        self.hPixel_real):
                                # cv2.line(img_mark, tuple(p1), tuple(p2), (255, 255, 0), 2)
                                cv2.line(img_mark, tuple(p1), tuple(p2), self.colour_dict8[2], 2)
                            if self.point_cheak(p2, self.wPixel_real, self.hPixel_real) and self.point_cheak(p3,
                                                                                                             self.wPixel_real,
                                                                                                             self.hPixel_real):
                                # cv2.line(img_mark, tuple(p1), tuple(p2), (255, 255, 0), 2)
                                cv2.line(img_mark, tuple(p2), tuple(p3), self.colour_dict8[2], 2)
                        # 画缝隙
                        if i < num_group - 1:
                            pp = points_all[(i + 1) * points_num_each_group + j * points_num_each_line]
                            if self.point_cheak(p4, self.wPixel_real, self.hPixel_real) and self.point_cheak(pp,
                                                                                                             self.wPixel_real,
                                                                                                             self.hPixel_real):
                                # cv2.line(img_mark, tuple(p1), tuple(p2), (255, 255, 0), 2)
                                cv2.line(img_mark, tuple(p4), tuple(pp), self.colour_dict8[3], 2)
        # # 按照顺序连接点  中间没有间断
        if True:
            lines_list = [lines1, lines2, lines3, lines4]
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
            try:
                # 画煤
                if len(lines1) > 0 and len(lines2) > 0:
                    points_coals = np.concatenate((lines1, lines2[::-1]), axis=0)
                    # 填充多边形
                    cv2.fillPoly(self.count_like_label, [points_coals], (0, 0, 255))
                    cv2.fillPoly(self.count_like_label_np, [points_coals], self.list_class.index('coal'))
                # 画传送带
                if len(lines2) > 0 and len(lines3) > 0:
                    points_transmissionBelts = np.concatenate((lines2, lines3[::-1]), axis=0)
                    # 填充多边形
                    cv2.fillPoly(self.count_like_label, [points_transmissionBelts], (181, 119, 53))
                    cv2.fillPoly(self.count_like_label_np, [points_transmissionBelts],
                                self.list_class.index('transmissionBelt'))
            except Exception as e:
                print('err608',e)
        return img_mark
    def mark_4point(self, points_all, img_mark=None,flag_only_plates_dict=False):
        up_y = 50
        low_y = 100
        points = []
        plates_dict = {}
        points_num_each_line = int(len(self.plate_Word) / 2)
        plate_id = -int((len(points_all) / len(self.plate_Word) - 1) / 2)
        for point in points_all:
            points.append(point)
            if len(points) == len(self.plate_Word):
                plate_point_ok_index = set()
                right_point_num = self.points_cheak_right_num(points, self.wPixel_real, self.hPixel_real, 0.5)
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
                        points_Move_Up = [np.array(
                            [arr[0] if i < len(points) / 2 else points[int(len(points) - i - 1)][0],
                             up_y if i < len(points) / 2 else low_y], dtype=np.int16) for i, arr in enumerate(points)]

                        points_Move_Up = points_Move_Up if self.flag_plate_UP else points
                        # 画真实的护帮板
                        if True:
                            if img_mark is not None and not flag_only_plates_dict:
                                # 画中心点
                                if right_point_num == len(points) and abs(plate_id) < 5:
                                    x_draw_plate = int(sum([row[0] for row in points_Move_Up]) / len(points))
                                    y_draw_plate = int(sum([row[1] for row in points_Move_Up]) / len(points))
                                    pixel_draw_row = 20
                                    # print(str(plate_id),'w'+str(points_Move_Up[1][0]-points_Move_Up[0][0]),'h'+str(points_Move_Up[3][1] - points_Move_Up[0][1]))
                                    if self.point_cheak((x_draw_plate, y_draw_plate), self.wPixel_real, self.hPixel_real):
                                        # 在图片上添加文本标记序号
                                        # 画架号
                                        if True:
                                            text = str(plate_id + int(self.ID))
                                            # text = str(plate_id)
                                            font, font_scale, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                                            # 获取文本尺寸
                                            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale,font_thickness)

                                            # 计算文本的起始点，使文本的中心与指定点对齐
                                            origin_x = x_draw_plate - text_width // 2
                                            origin_y = y_draw_plate + text_height // 2

                                            cv2.putText(img_mark, text, (origin_x, origin_y), font, font_scale,
                                                        self.colour_dict8[1], font_thickness)
                                        # 显示宽度
                                        if False:
                                            cv2.putText(img_mark, 'w' + str(points_Move_Up[2][0] - points_Move_Up[0][0]),
                                                        (x_draw_plate, y_draw_plate + pixel_draw_row),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colour_dict8[0], 1)
                                            cv2.putText(img_mark, 'w' + str(points_Move_Up[3][0] - points_Move_Up[3][0]),
                                                        (x_draw_plate, y_draw_plate + pixel_draw_row * 2),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colour_dict8[0], 1)
                                            cv2.putText(img_mark, 'h' + str(points_Move_Up[5][1] - points_Move_Up[0][1]),
                                                        (x_draw_plate, y_draw_plate + pixel_draw_row * 3),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colour_dict8[0], 1)
                                            cv2.putText(img_mark, 'h' + str(points_Move_Up[3][1] - points_Move_Up[1][1]),
                                                        (x_draw_plate, y_draw_plate + pixel_draw_row * 4),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colour_dict8[0], 1)

                                # 标记点的序号
                                if False:
                                    for i, point in enumerate(points_Move_Up):
                                        if self.point_cheak(point, self.wPixel_real, self.hPixel_real) and i < 4:
                                            x, y = point
                                            # 在图片上画点
                                            cv2.circle(img_mark, (x, y), 5, (0, 255, 0), -1)
                                            # 在图片上添加文本标记序号
                                            cv2.putText(img_mark, str(i), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255), 1)
                            # 按照顺序连接点 护帮板
                            if True:
                                # if abs(plate_id) < 5:
                                    points_Move_Up.append(points_Move_Up[0])
                                    for i in range(len(points_Move_Up) - 1):
                                        p1 = points_Move_Up[i]
                                        p2 = points_Move_Up[i + 1]
                                        if self.point_cheak(p1, self.wPixel_real,self.hPixel_real) and self.point_cheak(p2, self.wPixel_real,self.hPixel_real):
                                            # cv2.line(img_mark, tuple(p1), tuple(p2), (255, 255, 0), 2)
                                            if img_mark is not None and not flag_only_plates_dict:
                                                cv2.line(img_mark, tuple(p1), tuple(p2), (255, 0, 0), 2)
                                            plate_point_ok_index.add(i)
                                            plate_point_ok_index.add(i+1)
                                    plates_dict[plate_id] = [points_Move_Up[idex] for idex in sorted(plate_point_ok_index)]
                            # 画两条等高线
                            if False:
                                contour_line_1 = 0.6
                                contour_line_2 = 0.4
                                if self.point_cheak(points):
                                    for i in range(len(points) - 1):
                                        ph11 = (int(points[0][0] + contour_line_1 * (points[9][0] - points[0][0])),
                                                int(points[0][1] + contour_line_1 * (points[9][1] - points[0][1])))
                                        ph12 = (int(points[4][0] + contour_line_1 * (points[5][0] - points[4][0])),
                                                int(points[4][1] + contour_line_1 * (points[5][1] - points[4][1])))
                                        if self.point_cheak(ph11, self.wPixel_real,self.hPixel_real) and self.point_cheak(ph12,self.wPixel_real,self.hPixel_real):
                                            # cv2.line(img_mark, tuple(p1), tuple(p2), (255, 255, 0), 2)
                                            cv2.line(img_mark, tuple(ph11), tuple(ph12), (0, 0, 255), 2)
                                        ph21 = (int(points[0][0] + contour_line_2 * (points[9][0] - points[0][0])),
                                                int(points[0][1] + contour_line_2 * (points[9][1] - points[0][1])))
                                        ph22 = (int(points[4][0] + contour_line_2 * (points[5][0] - points[4][0])),
                                                int(points[4][1] + contour_line_2 * (points[5][1] - points[4][1])))
                                        if self.point_cheak(ph21, self.wPixel_real,self.hPixel_real) and self.point_cheak(ph22,self.wPixel_real,self.hPixel_real):
                                            # cv2.line(img_mark, tuple(p1), tuple(p2), (255, 255, 0), 2)
                                            cv2.line(img_mark, tuple(ph21), tuple(ph22), (0, 255, 0), 2)
                        # #画投影的护帮板 给杨哥的返回值(只转这个)
                        if img_mark is not None:
                            if abs(plate_id)<5:
                                # points_ = [[p[0],100] for p in points]
                                points_ = points
                                top_left, bottom_right = None, None
                                for i in range(points_num_each_line):
                                    if self.point_cheak(points_[i], self.wPixel_real, self.hPixel_real, 0):
                                        top_left = [points[i][0], up_y]
                                        break
                                for j in range(points_num_each_line - 1, 0, -1):
                                    if self.point_cheak(points_[j], self.wPixel_real, self.hPixel_real, 0):
                                        bottom_right = [points[j][0], low_y]
                                        break
                                if (top_left is not None) and (bottom_right is not None) and j - i >= 1:
                                    # print('i,j',i,j)
                                    point_left_upper_right_lower = [top_left, bottom_right]
                                    return_result = point_left_upper_right_lower, plate_id
                                    # 画矩形 虚拟架号
                                    cv2.rectangle(img_mark, top_left, bottom_right, (0, 255, 0), 2)
                                    # 计算矩形中心点
                                    center_x = (top_left[0] + bottom_right[0]) // 2
                                    center_y = (top_left[1] + bottom_right[1]) // 2
                                    # text = str(plate_id)
                                    numbers = int(''.join(char for char in self.ID if char.isdigit()))
                                    # print(numbers)  # 输出：33
                                    text = str(plate_id + numbers)
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 1
                                    font_thickness = 2
                                    # 获取文本的宽度和高度
                                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale,font_thickness)
                                    # 计算文本的左下角坐标（OpenCV的putText函数要求的）
                                    text_x = center_x - text_width // 2
                                    text_y = center_y + text_height // 2
                                    cv2.putText(img_mark, text, (text_x, text_y), font, font_scale, (0, 255, 0),font_thickness)
                        # 画真实的护帮板,画到底板，算损失用
                        if True:
                            points_transmissionBelts = np.array([point for point in points if self.point_cheak(point, self.wPixel_real,self.hPixel_real, 0.5)]).astype(np.int32)
                            if len(points_transmissionBelts) > 2:
                                try:
                                    points_transmissionBelts = points_transmissionBelts[:, np.newaxis, :]
                                    # cv2.drawContours(img_mark, [points_transmissionBelts], -1, (85, 255, 0), -1)
                                    cv2.drawContours(self.count_like_label, [points_transmissionBelts], -1, (0, 255, 85),-1)
                                    cv2.drawContours(self.count_like_label_np, [points_transmissionBelts], -1,
                                                    self.list_class.index('plates'), -1)
                                except Exception as e:
                                    print('err770',e)
                points = []
                plate_id += 1
        return img_mark,plates_dict
    def single_mark_lines_palte_fold(self,points, img_mark):
        new_points_index = set()
        # colour = self.colour_dict16[8]
        colour = self.colour_dict8[2]
        # 按照顺序连接点
        if True:
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                # if self.point_cheak(p1, self.wPixel_real, self.hPixel_real) and self.point_cheak(p2,self.wPixel_real,self.hPixel_real):
                    # cv2.line(img_mark, tuple(p1), tuple(p2), (255, 255, 0), 2)
                if True:
                    cv2.line(img_mark, tuple(p1), tuple(p2), colour, 2)
                    new_points_index.add(i)
                    new_points_index.add(i+1)
        return img_mark
    def single_mark_lines(self, points_all, img_mark, ids):
        num_group = len(ids)
        points_num_each_group = int(len(points_all) / num_group)
        for id_index in range(num_group):
            points = points_all[id_index * points_num_each_group:(id_index + 1) * points_num_each_group]
            right_point_num = self.points_cheak_right_num(points, self.wPixel_real, self.hPixel_real, 0.5)
            colour = self.colour_dict16[ids[id_index] % len(self.colour_dict16)]
            if False:
                if ids[id_index] != 4 or id_index % 2 == 0:
                    continue
                if right_point_num == 0:
                    pass
                if right_point_num < 4:
                    pass
            else:
                # 按照顺序连接点
                if True:
                    for i in range(len(points) - 1):
                        p1 = points[i]
                        p2 = points[i + 1]
                        if self.point_cheak(p1, self.wPixel_real, self.hPixel_real) and self.point_cheak(p2,self.wPixel_real,
                                                                                                         self.hPixel_real):
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
                    # text = str(ids[id_index]+int(self.ID))
                    text = str(ids[id_index])
                    font, font_scale, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    k = 0 if id_index % 2 == 0 else -1
                    x, y = points[k]
                    cv2.putText(img_mark, text, (x, y), font, font_scale, colour, font_thickness)
            if True:  # 将同一个支架连接起来
                if id_index % 2 == 0:
                    p1 = points_all[id_index * points_num_each_group]
                    p2 = points_all[(id_index + 1) * points_num_each_group]
                    if self.point_cheak(p1, self.wPixel_real, self.hPixel_real) and self.point_cheak(p2,self.wPixel_real,self.hPixel_real):
                        cv2.line(img_mark, tuple(p1), tuple(p2), colour, 2)
        return img_mark
    def Filter_points(self,points,wPixel_real, hPixel_real):
        new_points = []
        for p in points:
            if self.point_cheak(p, wPixel_real, hPixel_real):
                new_points.append(p)
        return new_points
    # 需要转
    def count(self, Tilt, Pan, image1, np_predict=None):
        self.hPixel_real, self.wPixel_real, _ = image1.shape
        points_4d,points_plate_fold,points_line,points_boundary = self.count_no_img(Tilt, Pan,self.wPixel_real, self.hPixel_real)
        self.count_like_label = np.zeros([self.hPixel_real, self.wPixel_real, 3]).astype(np.uint8)
        self.count_like_label_np = np.zeros([self.hPixel_real, self.wPixel_real]).astype(np.uint8)
        img_mark = image1
        if self.flag_test and np_predict is not None:
            mask = (np_predict == self.list_class.index('plates')) | (
                    np_predict == self.list_class.index('transmissionBelt')) | (
                           np_predict == self.list_class.index('coal'))
            np_predict[~mask] = 0
            if self.np_predicts is None:
                self.np_predicts = np_predict[None, :, :]
            else:
                try:
                    self.np_predicts = np.vstack((self.np_predicts, np_predict[None, :, :]))
                except Exception as e:
                    # self.np_predicts None
                    self.flag_vstack = False
                    print('err996', e)
        img_mark,plates_dict = self.mark_4point(points_4d, img_mark)
        img_mark = self.mark_lines(points_line, img_mark)
        #画护帮板折叠线,干涉，采煤机
        # img_mark = self.single_mark_lines_palte_fold(points_plate_fold, img_mark)
        ids = self.boundaries_id
        ids = [item for item in ids for _ in range(2)]
        # # 需要转240711
        img_mark = self.single_mark_lines(points_boundary, img_mark, ids)
        if self.count_like_label_nps is None:
            self.count_like_label_nps = self.count_like_label_np[None, :, :]
        else:
            try:
                self.count_like_label_nps = np.vstack((self.count_like_label_nps, self.count_like_label_np[None, :, :]))
            except Exception as e:
                print('err1027', e)
        return img_mark, self.count_like_label, self.count_like_label_np,points_plate_fold
    # 需要转
    def count_no_img(self, Tilt, Pan,wPixel_real, hPixel_real):
        '''
        第一个参数 上下转（绕着x轴转）  单位：角度  该值为相对值，正对煤比为0，以下为正。
        第二个参数 左右转（绕着y轴转）  单位：角度  该值为相对值，正对煤比为0，以左为-，以右为+
        '''
        Pan = -Pan
        # Tilt = 0
        # Pan = 0
        if False:#euler
            if False:
                self.RT = self.RT_count2(Tilt + self._T_constant_camera, Pan)
                self.RT_support = self.RT_count2(Tilt + self._T_constant_support + self._T_constant_camera, Pan)
            else:#ok
                self.RT = self.evler.RT_count3(Tilt + self._T_constant_camera, Pan)
                self.RT_support = self.evler.RT_count3(Tilt + self._T_constant_support + self._T_constant_camera, Pan)
            if False:
                w_4d = self.all_points_Word[:len(self.plates_Word)]
                w_line = self.all_points_Word[len(self.plates_Word):len()]

                points_4d = self.World_pixel(w_4d, self.RT, self.K, )
                points_line = self.World_pixel(w_line, self.RT_support, self.K, )
            elif True: #ok Euler
                self.evler.hPixel_real,self.evler.wPixel_real = self.hPixel_real, self.wPixel_real
                # 需要转
                points_4d = self.evler.World_pixel(self.plates_Word, self.RT, self.evler.K, )

                points_plate_fold = self.evler.World_pixel(self.plates_fold_Word, self.RT, self.evler.K, )
                # 不需要转
                points_line = self.evler.World_pixel(self.transmissionBelts_Word, self.RT_support, self.evler.K, )
                # 需要转
                points_boundary = self.evler.World_pixel(self.boundaries_Word, self.RT_support, self.evler.K, )
        else:#siyuan
            # 需要转
            # Pan = -Pan
            Tilt = -Tilt
            '''
            右转为正    yaw,    Pan
            上转为正    pitch   Tilt
            '''
            # self.World_pixel_siyuan(self.plates_Word, 0, 0,wPixel_real, hPixel_real,False)
            points_4d = self.World_pixel_siyuan(self.plates_Word, Tilt, Pan,wPixel_real, hPixel_real)
            points_plate_fold = self.World_pixel_siyuan(self.plates_fold_Word,Tilt, Pan,wPixel_real, hPixel_real)
            # 不需要转
            points_line = self.World_pixel_siyuan(self.transmissionBelts_Word, Tilt, Pan,wPixel_real, hPixel_real)
            # 需要转
            points_boundary = self.World_pixel_siyuan(self.boundaries_Word, Tilt, Pan,wPixel_real, hPixel_real )
        points_plate_fold = self.Filter_points(points_plate_fold,wPixel_real, hPixel_real)
        return points_4d,points_plate_fold,points_line,points_boundary
    def World_pixel_siyuan(self, points,Tilt, Pan,wPixel_real, hPixel_real,flag_filter = True):
        camera = QuaternionCamera()
        # view_transform = ViewTransform(800, 600)  # 800x600分辨率
        view_transform = ViewTransform(wPixel_real, hPixel_real)  # 800x600分辨率
        camera.rotate(Pan, Tilt)
        # camera.rotate(0, 0)
        transform_point_to_screen([0,0,1], camera, view_transform,flag_filter)
        # 存储不同视角下点的屏幕坐标
        screen_positions = []
        for point in points:
            screen_pos = transform_point_to_screen(point, camera, view_transform,flag_filter)
            screen_positions.append(screen_pos)
        screen_positions = np.round(screen_positions).astype(np.int16)
        del camera
        del view_transform
        return screen_positions
    def count_loss(self, loss_name_path=None):
        self.print_jr('self.cam_num_plates,self.cam_num_transmissionBelts', self.cam_num_plates,
                      self.cam_num_transmissionBelts)
        if False:
            inputs = self.np_predicts
            targets = self.count_like_label_nps
            Class_IoU_new = None
            C = 8
            loss_jr = Loss_jr(inputs, targets, Class_IoU_new, C)  # 此处的targets 就有边缘轮廓渐变的现象发生
            # findContours.Pass_ginseng(inputs,targets)
            loss_jr.jrFindContours()
            loss_mysef = loss_jr.loss_point_num()
            print('loss_mysef', loss_mysef)
        else:
            if True:  # 不要coal 的区域算损失
                self.np_predicts[self.np_predicts == self.list_class.index('coal')] = 0
                self.count_like_label_nps[self.count_like_label_nps == self.list_class.index('coal')] = 0

            if 'all' not in self.flag_calculation_depends:
                calculation_npt_depends = set(self.list_class) - set(self.flag_calculation_depends) - set('background')
                for calculation_npt_depend in calculation_npt_depends:  # calculation_npt_depend  的区域算损失
                    self.np_predicts[self.np_predicts == self.list_class.index(calculation_npt_depend)] = 0
                    self.count_like_label_nps[self.count_like_label_nps == self.list_class.index(calculation_npt_depend)] = 0
            else:
                pass
            # 使用 == 运算符比较两个数组
            equal_elements = self.np_predicts == self.count_like_label_nps
            # equal_elements = self.np_predicts == self.np_predicts
            # 使用 numpy.sum() 统计 True 的个数
            count_equal_elements = np.sum(equal_elements)
            Equal_proportion_elements = count_equal_elements / np.size(equal_elements) * 100

            print(f"相等的元素个数比例: {Equal_proportion_elements} %")
            loss = 100 - Equal_proportion_elements

            if loss_name_path is not None:
                assert len(loss_name_path) == equal_elements.shape[0]
                equal_elements_ = (equal_elements * 255).astype(np.uint8)
                for i in range(len(loss_name_path)):
                    path_r = loss_name_path[i] + '_loss.jpg'
                    image = equal_elements_[i]
                    cv2.imwrite(path_r, image)
                    # assert  'ID68_T00_P279' not in path_r 
                print('len(loss_name_path)', len(loss_name_path))
                assert  1 ==2
        return loss
    def print_jr(self, *args):
        names = args[0].split(',')
        for i in range(len(names)):
            print(names[i], ' : ', args[i + 1])
    def main(self, root_path):
        # self.cam_num_transmissionBelts = 4 #12# 30#10#4 #100
        self.cam_num_transmissionBelts = 10  # 6#10
        self.cam_num_plates = 10#4  # 0#4
        self.init_jr()
        self.root_path = root_path
        result_path = root_path + '_result'
        result_path_label = root_path + '_label'
        for path in [result_path,result_path_label]:
            if not os.path.exists(path):
                os.mkdir(path)
        Vpath_aim = os.path.join(result_path, f'result_ID{self.ID}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video = cv2.VideoWriter(Vpath_aim, fourcc, 0.5, (self.wPixel_real, self.hPixel_real))
        video = None
        # camera_Principles = Camera_Principles()
        # 获取文件和目录列表
        files_and_dirs = os.listdir(root_path)
        img_paths = []
        for name in files_and_dirs:
            if '_predict' in name:
                continue
            elif '.mp4' in name:
                continue
            elif not str(self.ID) in name:
                continue
            img_paths.append(name)
        # 使用sorted函数按照文件名排序
        sorted_files_and_dirs = sorted(img_paths)
        # 输出排序后的列表
        for name in sorted_files_and_dirs:
            print('name ', name)
            # match = re.search(r'ID(\d+)_T(\d{2})_P(\d+).*.jpg', name)
            # match = re.search(r'ID([a-zA-Z]+\d+)_T(\d{2})_P(\d+).*.jpg', name)
            match = re.search(r'ID(\w+)_T(\d{2})_P(\d+).*.jpg', name)
            ID = str(match.group(1))
            Tilt = int(match.group(2))
            Pan = int(match.group(3))
            Pan = Pan - 180
            if str(self.ID) != str(ID):
                continue
            image1 = cv2.imread(os.path.join(root_path, name))
            self.hPixel_real, self.wPixel_real, _ = image1.shape
            if video is None:
                video = cv2.VideoWriter(Vpath_aim, fourcc, 0.5, (self.wPixel_real, self.hPixel_real))
            # predict_npy_path = os.path.join(root_path, name.split('.')[0] + '_predict_solo.npy')
            predict_npy_path = os.path.join(root_path, name.split('.')[0] + '_predict.npy')
            if self.flag_test:
                try:
                    # np_predict = np.load()
                    np_predict = np.load(predict_npy_path)
                except:
                    np_predict = None
                    print(predict_npy_path, '不存在')
            # Tilt, Pan, = 15,30
            # Tilt, Pan, = 30, -90
            # 需要转
            # # 注意：这会导致像素值的简单相加，可能导致溢出，你可能需要对结果进行归一化
            # result = cv2.addWeighted(image1, 0.5, img_mark,0.5, 0)
            if self.flag_test and (np_predict is not None):  # Dan_pat
                np_predict[np_predict == self.list_class.index('plates')] = 0
            if self.flag_test:
                img_mark, img_mark_label, count_like_label_np,points_plate_fold = self.count(Tilt, Pan, image1, np_predict)
                result = img_mark
                # 保存结果图片
                path_r = os.path.join(result_path, name.split('.')[0] + '_T' + str(Tilt) + '_P' + str(Pan))
                cv2.imwrite(path_r + '.jpg', result)
                if np_predict is not None:
                    cv2.imwrite(path_r + '_img_mark_label.png', img_mark_label)
                    cv2.imwrite(path_r + '_count_like_label_np.png', count_like_label_np * 30)
                    cv2.imwrite(path_r + '_np_predict.png', np_predict * 30)
                    # 使用 == 运算符比较两个数组
                    # 检查形状是否匹配
                    if count_like_label_np.shape == np_predict.shape:
                        # equal_element = count_like_label_np == np_predict
                        pass
                    else:
                        print("Arrays do not have the same shape.")
                    equal_element = count_like_label_np == np_predict
                    cv2.imwrite(os.path.join(path_r + '_np_equal_element.png'), equal_element * 255)
                video.write(result)
            else:
                self.count(Tilt, Pan, image1)
        if self.flag_test:
            if video is not None:
                video.release()  # 释放
            try:
                loss = self.count_loss()
                print('loss', loss)
                return loss
            except:
                print('err loss')
        return points_plate_fold
    def calculate_error(self, ):
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
            Pan = Pan - 180
            if str(self.ID) != str(ID):
                continue
            # predict_npy_path = os.path.join(root_path, name.split('.')[0] + '_predict_solo.npy')
            predict_npy_path = os.path.join(root_path, name.split('.')[0] + '_predict.npy')
            # np_predict = np.load(os.path.join(self.root_path, name.split('.')[0] + '_predict.npy'))
            np_predict = np.load(predict_npy_path)
            image1 = cv2.imread(os.path.join(self.root_path, name))
            self.count(Tilt, Pan, image1, np_predict)
            path_r = os.path.join(result_path, name.split('.')[0] + '_T' + str(Tilt) + '_P' + str(Pan))
            loss_name_path.append(path_r)
        if self.flag_vstack:
            loss = self.count_loss(loss_name_path)
        else:
            loss = None
        self.error_function_N0 += 1
        print('\nID:', self.ID, ' loss:', loss, self.error_function_N0, '平均花费时间：',
              (time.time() - self.time_begin) / self.error_function_N0)
        return loss
    # 自定义函数，计算误差
    def error_function(self, params):
        # params是一个包含5个未知参数的数组
        # 参数变少，时间减少是指数级的
        if 'all' in self.flag_calculation_depends:
            self.d_level, self.installHeight, self.transmissionBelt_Coal_wall1, self._T_constant_camera, self._T_constant_support = params
        elif 'plates' in self.flag_calculation_depends and len(self.flag_calculation_depends) == 1:
            self.d_level, self._T_constant_camera = params
        elif 'transmissionBelt' in self.flag_calculation_depends and len(self.flag_calculation_depends) == 1:
            self.installHeight, self.transmissionBelt_Coal_wall1, self._T_constant_support = params
        else:
            self.installHeight, self.transmissionBelt_Coal_wall1, self._T_constant_support = params
        self.init_jr()
        # self.print_jr('self.flag_calculation_depends,self.d_level, self.installHeight, self.transmissionBelt_Coal_wall1,self._T_constant_camera,
        # self._T_constant_support',self.flag_calculation_depends,self.d_level, self.installHeight, self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support,)
        t_temp = time.time()
        # 这里替换为误差计算逻辑
        # 假设 calculate_error 是计算误差的函数
        error = self.calculate_error()
        print('单次花费时间：', time.time() - t_temp)
        return error

    def minimize_count(self):
        self.cam_num_transmissionBelts = 10  # 6#10
        self.cam_num_plates = 8  # 20
        self.init_jr()
        # 初始猜测值，可以是任意合法的初始参数值
        # initial_guess = np.zeros(5)  # 例如，全部设为0
        # initial_guess = np.array([self.d_level,self.installHeight,self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support]).astype(np.float16)
        # initial_guess = np.array([self.d_level,self.installHeight,self.transmissionBelt_Coal_wall1,self._T_constant_camera,self._T_constant_support]).astype(np.int32)
        print('self.flag_calculation_depends', self.flag_calculation_depends)
        if 'all' in self.flag_calculation_depends:
            params = self.d_level, self.installHeight, self.transmissionBelt_Coal_wall1, self._T_constant_camera, self._T_constant_support
            list_temp = [[1e3, 10e3], [1e3, 5e3], [1e3, 5e3], [-20, 20], [-20, 20]]
        elif 'plates' in self.flag_calculation_depends and len(self.flag_calculation_depends) == 1:
            params = self.d_level, self._T_constant_camera
            list_temp = [[1e3, 10e3], [-20, 20]]
        elif 'transmissionBelt' in self.flag_calculation_depends and len(self.flag_calculation_depends) == 1:
            params = self.installHeight, self.transmissionBelt_Coal_wall1, self._T_constant_support
            list_temp = [[1e3, 5e3], [1e3, 5e3], [-20, 20]]
        else:
            params = self.installHeight, self.transmissionBelt_Coal_wall1, self._T_constant_support
            list_temp = [[1e3, 5e3], [1e3, 5e3], [-20, 20]]
        initial_guess = params
        # 约束
        # 边界约束（可选）
        bounds_mins = [l[0] for l in list_temp]
        bounds_maxs = [l[1] for l in list_temp]
        bounds = Bounds(bounds_mins, bounds_maxs)
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
            'maxiter': 100,  # 50,     # 最大迭代次数 100
            'maxfun': 50,  # 最大函数评估次数 150
            'ftol': 1e-6,  # 函数值容差
            'xtol': 1e-6,  # 参数值容差
            'gtol': 1e-6,  # 梯度容差
            'disp': True  # 打印优化过程
        }
        # 使用不同的方法进行优化
        methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr']
        # L-BFGS-B 不收敛
        # COBYLA 变化慢
        # 使用Nelder-Mead法进行优化
        result = minimize(self.error_function, initial_guess, method='nelder-mead',bounds=bounds, options=options)#constraints=cons
        # result = minimize(self.error_function, initial_guess, method='COBYLA', bounds=bounds,options=options)  # constraints=cons
        # 输出最优解
        print("最优参数:", result.x)
        print("最小误差:", result.fun)
        guess = np.array([self.d_level, self.installHeight, self.transmissionBelt_Coal_wall1, self._T_constant_camera,
                          self._T_constant_support]).astype(np.float16)
        np.save(self.name_initial_guess, guess)  # 保存文件
        print(f'#{self.ID}保存参数成功')
    def point_Calculate_angle(self, point):
        x, y, z = point
        r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        a_LR = math.degrees(math.atan(x / z))
        a_UP = math.degrees(math.atan(y / z))
        return a_UP, a_LR
    def quadrilateral_centroid(self, vertices):
        x_coords = [vertex[0] for vertex in vertices]
        y_coords = [vertex[1] for vertex in vertices]
        z_coords = [vertex[2] for vertex in vertices]
        centroid_x = sum(x_coords) / len(vertices)
        centroid_y = sum(y_coords) / len(vertices)
        centroid_z = sum(z_coords) / len(vertices)
        return centroid_x, centroid_y, centroid_z
    def Calculate_degrees_see(self, points):
        center = self.quadrilateral_centroid(points)
        a_UP, a_LR = self.point_Calculate_angle(center)
        RT = self.RT_count(a_UP, a_LR)
        points_region = self.World_pixel(points, RT, self.K, )
        right_point_num = self.points_cheak_right_num(points_region, self.wPixel_referenc, self.hPixel_referenc, 0.5)
        if right_point_num == len(points):
            print('该相机可以看全', right_point_num, len(points))
        else:
            print('该相机看不全', right_point_num, len(points))
        print('a_UP,a_LR', a_UP, a_LR)
        self.print_jr('points,points_region', points, points_region)
def long_time_task(ID, root_path):
    flag_minimize_count = True
    while flag_minimize_count:
        # 需要转
        camera_Principles = Camera_Principles(ID, root_path)
        # 需要转
        camera_Principles.main(root_path)
        del camera_Principles
        flag_minimize_count = False
        if platform.system() == 'Linux' or True:
            flag_minimize_count = True
            if False:
                # # 不需要转
                camera_Principles = Camera_Principles(ID, root_path)
                camera_Principles.flag_calculation_depends = ['plates']
                camera_Principles.minimize_count()
                del camera_Principles
            if False:
                # 不需要转
                camera_Principles = Camera_Principles(ID, root_path)
                camera_Principles.flag_calculation_depends = ['plates', 'transmissionBelt', ]
                camera_Principles.minimize_count()
                del camera_Principles
            if True:
                # 不需要转
                camera_Principles = Camera_Principles(ID, root_path)
                camera_Principles.flag_calculation_depends = ['all']
                camera_Principles.minimize_count()
                del camera_Principles
            if False:
                # 不需要转
                camera_Principles = Camera_Principles(ID, root_path)
                camera_Principles.flag_calculation_depends = ['transmissionBelt', ]
                camera_Principles.minimize_count()
                del camera_Principles
if __name__ == '__main__':
    if platform.system() == 'Linux':
        # input = r'/data/ins/dataset/test/img0705'
        root_path = r'/data/ins/dataset/test/all'
        # root_path = r'/data/ins/dataset/test/test2'
        # root_path = r'/data/ins/dataset/test/dif'
        # root_path = r'/data/ins/dataset/Virtual_plate_number/solov2/all_solo'
        # root_path = r'/data/ins/dataset/Virtual_plate_number/solov2/video_img'
        root_path = r'/data/ins/dataset/Virtual_plate_number/solov2/total_img'
    else:
        # root_path = r'F:\Desktop\move_ptz_img\angle_#68_result'
        # root_path = r'F:\Desktop\move_ptz_img\angle_#68'
        # root_path = r'F:\Desktop\move_ptz_img\angle_#44'
        # root_path = r'F:\Desktop\move_ptz_img\all'
        # root_path = r'F:\Desktop\move_ptz_img\all_result'
        # root_path = r'D:\dataset\Virtual_plate_number'
        # root_path = r'D:\dataset\Virtual_plate_number\all'
        # root_path = r'D:\dataset\Virtual_plate_number\solov2\all_solo'
        # root_path = r'D:\dataset\Virtual_plate_number\solov2\video_img'
        # root_path = r'D:\dataset\Virtual_plate_number\demo\all_solo'
        root_path = r'D:\dataset\Virtual_plate_number\deeplab\all_deeplab'
    # IDs = ['dhz33']
    # IDs = [68, 44]
    IDs = [68]
    # IDs = [68,44,103,118,999]
    # IDs = [103,118,999]
    # IDs = ['dhz33',44,68]

    if True:
        for ID in IDs:
            # # 创建子进程
            if False or platform.system() == 'Linux':
                p = Process(target=long_time_task, args=(ID, root_path))  # target进程执行的任务, args传参数（元祖）
                p.start()  # 启动进程
            else:
                long_time_task(ID, root_path)
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
