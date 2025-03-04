import math
import numpy as np
import cv2
from tools_jr import *
class Euler():
    def __init__(self,wPixel_referenc,hPixel_referenc,fx,fy,dist,flag_print_detal=True):
    
        self.flag_print_detal = flag_print_detal
        self.wPixel_referenc = wPixel_referenc
        self.hPixel_referenc = hPixel_referenc

        # self.fx, self.fy,self.k1, self.k2, self.p1, self.p2, self.k3 = Camera_internal_reference
        # self.dist = self.k1, self.k2, self.p1, self.p2, self.k3
        self.fx = fx
        self.fy = fy
        self.dist = dist
        self.k1, self.k2, self.p1, self.p2, self.k3 = self.dist

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


    def rotation_matrix_x(self, angle_radians):
        """
        返回绕X轴旋转的旋转矩阵
        """
        c = math.cos(angle_radians)
        s = math.sin(angle_radians)
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])

    def rotation_matrix_y(self, angle_radians):
        """
        返回绕Y轴旋转的旋转矩阵
        """
        c = math.cos(angle_radians)
        s = math.sin(angle_radians)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    def rotation_matrix_z(self, angle_radians):
        """
        返回绕Z轴旋转的旋转矩阵
        """
        c = math.cos(angle_radians)
        s = math.sin(angle_radians)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])

    def RT_count_ori(self, Tilt, Pan):
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

        if min(Tilt, Pan) < 30 and False:  # 官方
            # R_3x3 = np.dot(Rz, np.dot(Ry, Rx)).astype(np.float32)  # 先执行Ry和Rx的乘法，然后将结果与Rz相乘
            R_3x3 = Rz @ Ry @ Rx
        else:
            R_3x3 = Rz @ Rx @ Ry
        if False:
            self.t = np.array([tx, ty, tz], dtype=np.float32)
            # 将 t 转换为 3x1 矩阵
            t = self.t.reshape((3, 1))
            # 2.3 刚体变换：平移+旋转,平移矩阵和旋转结果相乘的结果
            # 合并 R 和 t 形成 3x4 的外参矩阵
            Rt = np.hstack((R_3x3, t)).astype(np.float32)

            if self.flag_print_detal:
                print('T', Rt)
            return Rt
        else:
            return R_3x3

    def R_Rt(self, R_3x3):
        ## 2.1 平移
        tx = 0
        ty = 0
        tz = 0

        self.t = np.array([tx, ty, tz], dtype=np.float32)
        # 将 t 转换为 3x1 矩阵
        t = self.t.reshape((3, 1))
        # 2.3 刚体变换：平移+旋转,平移矩阵和旋转结果相乘的结果
        # 合并 R 和 t 形成 3x4 的外参矩阵
        Rt = np.hstack((R_3x3, t)).astype(np.float32)

        if self.flag_print_detal:
            print('T', Rt)
        return Rt

    def RT_count2(self, Tilt, Pan):
        # Pan = -Pan
        # 2.相机坐标系
        ## 2.1 平移
        tx = 0
        ty = 0
        tz = 0

        p = math.radians(Tilt)
        t = math.radians(Pan)

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
            [cos_t, sin_t * sin_p, sin_t * cos_p],
            [0, cos_p, -sin_p],
            [-sin_t, cos_t * sin_p, cos_t * cos_p]
        ])
        self.t = np.array([tx, ty, tz], dtype=np.float32)
        # 将 t 转换为 3x1 矩阵
        tt = self.t.reshape((3, 1))
        # 2.3 刚体变换：平移+旋转,平移矩阵和旋转结果相乘的结果
        # 合并 R 和 t 形成 3x4 的外参矩阵
        Rt = np.hstack((R_3x3, tt)).astype(np.float32)

        if self.flag_print_detal:
            print('T', Rt)
        return Rt

    def RT_count3(self, Tilt, Pan):
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

        self.t = np.array([tx, ty, tz], dtype=np.float32)
        # 将 t 转换为 3x1 矩阵
        tt = self.t.reshape((3, 1))
        # 2.3 刚体变换：平移+旋转,平移矩阵和旋转结果相乘的结果
        # 合并 R 和 t 形成 3x4 的外参矩阵

        Rt = np.hstack((R_3x3, tt)).astype(np.float32)
        if self.flag_print_detal:
            print('T', Rt)
        return Rt
    def K_count(self):
        # 3 投影矩阵,内参矩阵
        cx = self.wPixel_referenc / 2
        cy = self.hPixel_referenc / 2
        K = [[self.fx, 0, cx],
             [0, self.fy, cy],
             [0, 0, 1]
             ]
        self.K = np.array(K).astype(np.float32)
        if self.flag_print_detal:
            print('self.K', self.K)




    # 需要转
    def World_pixel(self, Worldsx3, Rt, K,flag_test=False):
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
        # space_ok = P_C[-1] > 1000
        space_ok = P_C[-1] > 0

        # 计算归一化图像平面坐标 (3xN)
        P_N = P_C[:3, :] / P_C[2, :]
        # 2.使用相机内参矩阵将归一化图像平面坐标转换为像素坐标 (3x3) * (3xN) = (3xN)
        P_pixel = K @ P_N

        # # 提取像素坐标 u, v
        # u = P_pixel[0, :]
        # v = P_pixel[1, :]
        P_pixel_f = P_pixel.T[:, :2]
        P_pixel_r = P_pixel_f

        k = 0.33  # 0.33 #0.35 #0.4 #0.4#0.2#0.3 #0.5就乱了

        # 创建一个布尔数组来标识满足条件的行
        satisfying_condition_f = (P_pixel_f[:, 0] >= -self.wPixel_referenc * k) & (
                P_pixel_f[:, 0] <= self.wPixel_referenc * (1 + k)) & (
                                         P_pixel_f[:, 1] >= -self.hPixel_referenc * k) & (
                                         P_pixel_f[:, 1] <= self.hPixel_referenc * (1 + k))

        # # 取反以标识不满足条件的行
        # not_satisfying_condition = ~satisfying_condition
        
        # # 使用 np.where 来找到不满足条件的行的索引
        # indices = np.where(not_satisfying_condition)[0]

        ok_index = space_ok & satisfying_condition_f

        if True:
            # 将旋转矩阵转换为旋转向量
            R = Rt[:, :3]
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

        satisfying_condition_r = (P_pixel_r[:, 0] >= -self.wPixel_referenc * k) & (
                P_pixel_r[:, 0] <= self.wPixel_referenc * (1 + k)) & (
                                         P_pixel_r[:, 1] >= -self.hPixel_referenc * k) & (
                                         P_pixel_r[:, 1] <= self.hPixel_referenc * (1 + k))
        ok_index = ok_index & satisfying_condition_r

        P_pixel_r = resolution_conversion(P_pixel_r,self.wPixel_real,self.wPixel_referenc,self.hPixel_real,self.hPixel_referenc)
        P_pixel_r = np.nan_to_num(P_pixel_r, copy=False, nan=-32768, posinf=-32768, neginf=-32768)

        P_pixel_r = np.round(P_pixel_r).astype(np.int16)

        P_pixel_r_ok = P_pixel_r.copy()

        P_pixel_r_ok[~ok_index] = [np.int16(-32768), np.int16(-32768)]

        # return P_pixel_r_ok,P_pixel_r
        if flag_test:
            return P_pixel_r
        else:
            return P_pixel_r_ok

