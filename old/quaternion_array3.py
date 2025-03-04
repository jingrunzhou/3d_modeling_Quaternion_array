import numpy as np
from math import sin, cos, sqrt, pi
from quaternion_array import Quaternion
from quaternion_array2 import ViewTransform
from quaternion_array2 import transform_point_to_screen

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


    