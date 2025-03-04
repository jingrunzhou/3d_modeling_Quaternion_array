import numpy as np
from math import atan2, asin, degrees, sqrt

from quaternion_array_ok import QuaternionCamera,ViewTransform,transform_point_to_screen

class CameraPositioner:
    def __init__(self, screen_width=800, screen_height=600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_center = np.array([screen_width/2, screen_height/2])
        
        # 相机初始设置
        self.camera = QuaternionCamera()
        self.view_transform = ViewTransform(screen_width, screen_height)
    
    def calculate_angles_to_center(self, target_point):
        """
        计算将目标点转到屏幕中心所需的相机角度
        target_point: 目标点的世界坐标 [x, y, z]
        返回: (yaw, pitch) 角度，单位为度
        """
        # 将点坐标转换为相对于相机的向量
        relative_pos = target_point - self.camera.position
        
        # 计算到点的距离
        distance = np.linalg.norm(relative_pos)
        
        # 标准化向量
        direction = relative_pos / distance
        
        # 计算偏航角（水平旋转）
        # atan2(z, x) 给出在xz平面上的角度
        yaw = degrees(atan2(direction[0], direction[2]))
        
        # 计算俯仰角（垂直旋转）
        # asin(y) 给出与xz平面的夹角
        pitch = degrees(asin(direction[1]))
        
        return -yaw, -pitch  # 注意角度取反，因为我们要相机转向点

    def center_point(self, point):
        """
        调整相机角度使点位于屏幕中心
        """
        # 计算需要的旋转角度
        yaw, pitch = self.calculate_angles_to_center(point)
        
        # 重置相机方向
        self.camera = QuaternionCamera()
        
        # 应用计算出的旋转
        self.camera.rotate(yaw, pitch)
        
        # 验证点是否在屏幕中心
        screen_pos = transform_point_to_screen(point, self.camera, self.view_transform)
        
        return {
            'camera_angles': (yaw, pitch),
            'screen_position': screen_pos,
            'target_position': point,
            'distance_from_center': np.linalg.norm(screen_pos - self.screen_center)
        }

def verify_centering():
    """
    验证点的居中效果
    """
    positioner = CameraPositioner(800, 600)
    target_point = np.array([1.0, 1.0, 1.0])
    
    # 1. 记录初始位置
    initial_screen_pos = transform_point_to_screen(
        target_point, 
        positioner.camera, 
        positioner.view_transform
    )
    
    # 2. 居中点
    result = positioner.center_point(target_point)
    
    # 3. 打印结果
    print("=== 点居中验证 ===")
    print(f"目标点世界坐标: {target_point}")
    print(f"初始屏幕位置: {initial_screen_pos}")
    print(f"需要的相机角度: Yaw={result['camera_angles'][0]:.2f}°, Pitch={result['camera_angles'][1]:.2f}°")
    print(f"居中后屏幕位置: {result['screen_position']}")
    print(f"屏幕中心位置: {positioner.screen_center}")
    print(f"距离屏幕中心的偏差: {result['distance_from_center']:.2f}像素")
    
    return result

# 执行验证
result = verify_centering()

# 定义一个函数来模拟点的动态移动
def simulate_point_movement():
    positioner = CameraPositioner(800, 600)
    movements = [
        np.array([1.0, 1.0, 1.0]),    # 初始位置
        np.array([2.0, 1.0, 1.0]),    # 向右移动
        np.array([2.0, 2.0, 1.0]),    # 向上移动
        np.array([2.0, 2.0, 2.0]),    # 向前移动


        np.array([0.0, 0.0, 1.0]),    # 
    ]
    
    results = []
    for point in movements:
        result = positioner.center_point(point)
        results.append(result)
        print(f"\n点移动到 {point}")
        print(f"需要的相机角度: Yaw={result['camera_angles'][0]:.2f}°, Pitch={result['camera_angles'][1]:.2f}°")
    
    return results

# 模拟点的移动
movement_results = simulate_point_movement()