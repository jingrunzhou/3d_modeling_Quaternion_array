import numpy as np
from math import sqrt, atan2, asin, degrees
from quaternion_array3 import QuaternionCamera
from quaternion_array2 import ViewTransform


class CameraDistortion:
    def __init__(self):
        # 径向畸变参数 (k1, k2, k3)
        self.radial_params = np.array([-0.28, 0.1, -0.02])
        
        # 切向畸变参数 (p1, p2)
        self.tangential_params = np.array([0.001, 0.001])
        
        # 相机内参矩阵
        self.camera_matrix = np.array([
            [800, 0, 400],    # fx, 0, cx
            [0, 800, 300],    # 0, fy, cy
            [0, 0, 1]         # 0, 0, 1
        ])
    
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
        
        return np.column_stack((x_distorted, y_distorted))
    



    
    def correct_distortion(self, distorted_points, max_iter=5):
        """
        使用迭代方法校正畸变
        distorted_points: Nx2 畸变后的图像坐标
        """
        points = distorted_points.copy()
        
        for _ in range(max_iter):
            # 计算当前点的畸变
            r2 = np.sum(points**2, axis=1)
            r4 = r2*r2
            r6 = r4*r2
            
            k1, k2, k3 = self.radial_params
            radial_factor = 1 + k1*r2 + k2*r4 + k3*r6
            
            p1, p2 = self.tangential_params
            x, y = points[:, 0], points[:, 1]
            
            x_tangential = 2*p1*x*y + p2*(r2 + 2*x*x)
            y_tangential = p1*(r2 + 2*y*y) + 2*p2*x*y
            
            # 更新估计的无畸变坐标
            x_corrected = (distorted_points[:, 0] - x_tangential) / radial_factor
            y_corrected = (distorted_points[:, 1] - y_tangential) / radial_factor
            
            points = np.column_stack((x_corrected, y_corrected))
            
        return points

class ImprovedCameraPositioner:
    def __init__(self, screen_width=800, screen_height=600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_center = np.array([screen_width/2, screen_height/2])
        
        self.camera = QuaternionCamera()
        self.view_transform = ViewTransform(screen_width, screen_height)
        self.distortion = CameraDistortion()
        
    def transform_point_with_distortion(self, point):
        """
        将3D点转换到屏幕空间，包含畸变处理
        """
        # 1. 转换到相机空间
        cam_matrix = self.camera.view_matrix()
        point_h = np.append(point, 1.0)
        point_camera = np.dot(cam_matrix, point_h)[:3]
        
        # 2. 投影到归一化平面
        if point_camera[2] != 0:
            normalized = point_camera[:2] / point_camera[2]
        else:
            normalized = point_camera[:2]
            
        # 3. 应用畸变
        distorted = self.distortion.apply_distortion(normalized.reshape(1, 2))
        
        # 4. 应用相机内参
        pixel_coords = np.dot(self.distortion.camera_matrix, 
                            np.append(distorted[0], 1))[:2]
        
        return pixel_coords
    

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

    
    def center_point_with_distortion(self, point):
        """
        考虑畸变的点居中计算
        """
        # 计算初始角度
        yaw, pitch = self.calculate_angles_to_center(point)
        
        # 应用旋转
        self.camera.rotate(yaw, pitch)
        
        # 获取畸变后的屏幕坐标
        screen_pos = self.transform_point_with_distortion(point)
        
        # 迭代优化以补偿畸变影响
        max_iterations = 5
        for i in range(max_iterations):
            # 计算到屏幕中心的偏差
            offset = screen_pos - self.screen_center
            
            # 如果偏差足够小，停止迭代
            if np.linalg.norm(offset) < 1.0:
                break
                
            # 根据偏差调整角度
            delta_yaw = -offset[0] * 0.01  # 转换系数
            delta_pitch = -offset[1] * 0.01
            
            # 应用微调
            self.camera.rotate(delta_yaw, delta_pitch)
            
            # 重新计算屏幕位置
            screen_pos = self.transform_point_with_distortion(point)
        
        return {
            'camera_angles': (yaw, pitch),
            'screen_position': screen_pos,
            'iterations': i + 1,
            'final_offset': np.linalg.norm(screen_pos - self.screen_center)
        }

def test_distortion_correction():
    """
    测试畸变矫正效果
    """
    positioner = ImprovedCameraPositioner()
    test_points = [
        np.array([1.0, 1.0, 1.0]),    # 对角线位置
        np.array([2.0, 0.0, 0.0]),    # 右侧位置
        np.array([0.0, 2.0, 0.0]),    # 上方位置
        np.array([0.0, 0.0, 2.0])     # 前方位置
    ]
    
    results = []
    for point in test_points:
        # 获取原始畸变位置
        original_pos = positioner.transform_point_with_distortion(point)
        
        # 居中处理
        result = positioner.center_point_with_distortion(point)
        
        print(f"\n测试点 {point}:")
        print(f"原始屏幕位置: {original_pos}")
        print(f"畸变矫正后位置: {result['screen_position']}")
        print(f"需要的相机角度: Yaw={result['camera_angles'][0]:.2f}°, "
              f"Pitch={result['camera_angles'][1]:.2f}°")
        print(f"迭代次数: {result['iterations']}")
        print(f"最终偏差: {result['final_offset']:.2f}像素")
        
        results.append(result)
    
    return results
if __name__ == '__main__':
    # 运行测试
    test_results = test_distortion_correction()