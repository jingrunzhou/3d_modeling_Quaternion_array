import numpy as np
from math import sin, cos, sqrt, pi
from quaternion_array import Quaternion
from quaternion_array2 import ViewTransform
from quaternion_array2 import transform_point_to_screen

import numpy as np


class CameraDistortion_f:
    def __init__(self):
        if False:
            # 径向畸变参数 (k1, k2, k3)
            self.radial_params = np.array([-0.28, 0.1, -0.02])
            
            # 切向畸变参数 (p1, p2)
            self.tangential_params = np.array([0.001, 0.001])



        elif False:
            self.dist = np.array([-4.23709796e-01, 2.23283964e-01, -6.57576014e-04, -4.92667899e-05, -6.80752617e-02],dtype=np.float32)

            # self.dist = np.zeros_like(self.dist)


            self.k1, self.k2, self.p1, self.p2, self.k3 = self.dist*0.00001
            # 径向畸变参数 (k1, k2, k3)
            self.radial_params = np.array([self.k1, self.k2, self.k3])
            
            # 切向畸变参数 (p1, p2)
            self.tangential_params = np.array([self.p1, self.p2])
        else:
            k1= -0.3527841
            k2= 0.1429235
            k3= -0.0198524

            # 切向畸变参数(p1, p2)  
            p1= 0.0012453
            p2= -0.0003891

            # 径向畸变参数 (k1, k2, k3)
            self.radial_params = np.array([k1, k2, k3])
            
            # 切向畸变参数 (p1, p2)
            self.tangential_params = np.array([p1, p2])











    def apply_distortion(self, normalized_points):
        """
        应用畸变模型到归一化坐标
        normalized_points: Nx2 归一化图像坐标
        """
        x = normalized_points[:, 0]
        y = normalized_points[:, 1]
        
        # 计算r^2, r^4, r^6
        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r4*r2
        
        # 径向畸变
        k1, k2, k3 = self.radial_params
        radial_factor = 1 + k1*r2 + k2*r4 + k3*r6
        
        # 切向畸变
        p1, p2 = self.tangential_params
        x_tangential = 2*p1*x*y + p2*(r2 + 2*x*x)
        y_tangential = p1*(r2 + 2*y*y) + 2*p2*x*y
        
        # 应用畸变
        x_distorted = x*radial_factor + x_tangential
        y_distorted = y*radial_factor + y_tangential


        # x_distorted = x*radial_factor
        # y_distorted = y*radial_factor
        
        return np.column_stack((x_distorted, y_distorted))
    





class QuaternionCamera:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 5.0])  # 相机位置
        self.orientation = Quaternion()  # 相机朝向
        self.target = np.array([0.0, 0.0, 0.0])  # 观察目标
        self.up = np.array([0.0, 1.0, 0.0])      # 上向量
        
    def rotate(self, yaw, pitch):
        """使用四元数进行相机旋转"""
        yaw_rotation = Quaternion.from_axis_angle(np.array([0, 1, 0]), yaw * pi / 180)
        pitch_rotation = Quaternion.from_axis_angle(np.array([1, 0, 0]), pitch * pi / 180)
        self.orientation = yaw_rotation.multiply(pitch_rotation.multiply(self.orientation))
        self.orientation.normalize()
        
        # 更新target位置（让相机始终看向前方）
        forward = np.array([0, 0, -1])
        rotation_matrix = self.orientation.to_matrix()
        forward = np.dot(rotation_matrix, forward)
        self.target = self.position + forward

    def view_matrix(self):
        """返回视图矩阵"""
        rotation = self.orientation.to_matrix()
        
        # 构建完整的视图矩阵
        view = np.eye(4)
        view[:3, :3] = rotation.T
        view[:3, 3] = -np.dot(rotation.T, self.position)
        
        return view


def main():
    # 创建相机和视口转换器
    camera = QuaternionCamera()
    view_transform = ViewTransform(800, 600)  # 800x600分辨率
    z0 = 1000.0
    # 定义一些3D点
    points = [
        np.array([0.0, 0.0, 0.0]),    # 原点
        np.array([1.0, 1.0, 0.0]),    # 右上前方
        np.array([-1.0, -1.0, -1.0]), # 左下后方

        np.array([0.0, 0.0, z0]),  
        np.array([z0, 0, z0]),    
        np.array([0, z0, z0]), 
        np.array([z0, z0, z0]), 

    ]
    


    # 存储不同视角下点的屏幕坐标
    screen_positions = []
    
    # 测试不同的视角
    rotations = [
        (0, 0),      # 初始视角
        (30, 0),     # 向右旋转30度
        (30, 20),    # 向右30度，向上20度
        (-45, -10),  # 向左45度，向下10度


        (0, 0),      # 初始视角
        (45, 0),      # 初始视角
        (0, 45),     # 向右旋转30度
        (45, 45),    # 向右30度，向上20度
        (-45, 45),    # 向右30度，向上20度
        (45, -45),    # 向右30度，向上20度
        (-45, -45),    # 向右30度，向上20度

    ]
    
    for yaw, pitch in rotations:
        # 旋转相机
        camera.rotate(yaw, pitch)
        
        # 计算每个点在屏幕上的位置
        frame_positions = []
        for point in points:
            screen_pos = transform_point_to_screen(point, camera, view_transform)
            frame_positions.append({
                'world_pos': point,
                'screen_pos': screen_pos,
                'camera_rotation': (yaw, pitch)
            })
        screen_positions.append(frame_positions)
        
        # 打印结果
        print(f"\n相机旋转: Yaw={yaw}°, Pitch={pitch}°")
        for pos in frame_positions:
            print(f"世界坐标 {pos['world_pos']} -> 屏幕坐标 {pos['screen_pos']}")
    
    return screen_positions









if __name__ == '__main__':
    # 运行测试
    if False:
        screen_coordinates = main()
    elif True:

        camera = QuaternionCamera()
        view_transform = ViewTransform(1280, 960)  # 800x600分辨率



        camera.rotate(yaw, pitch)
        screen_pos = transform_point_to_screen(point, camera, view_transform)


    