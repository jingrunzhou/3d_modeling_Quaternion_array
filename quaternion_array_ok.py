from math import sin, cos, sqrt, pi
import numpy as np
# from jibianjiaozhen import ImprovedCameraPositioner
from fun_jibian import CameraDistortion_f


class Quaternion:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w = w  # 实部
        self.x = x  # 虚部i
        self.y = y  # 虚部j
        self.z = z  # 虚部k
        self.normalize()

    def normalize(self):
        """标准化四元数"""
        norm = sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)
        if norm > 0:
            self.w /= norm
            self.x /= norm
            self.y /= norm
            self.z /= norm

    def multiply(self, other):
        """四元数乘法"""
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w, x, y, z)

    def conjugate(self):
        """四元数共轭"""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    @staticmethod
    def from_axis_angle(axis, angle):
        """从旋转轴和角度创建四元数"""
        half_angle = angle * 0.5
        sin_half = sin(half_angle)
        axis_normalized = axis / np.linalg.norm(axis)
        return Quaternion(
            cos(half_angle),
            axis_normalized[0] * sin_half,
            axis_normalized[1] * sin_half,
            axis_normalized[2] * sin_half
        )

    def to_matrix(self):
        """转换为旋转矩阵"""
        w2, x2, y2, z2 = self.w*self.w, self.x*self.x, self.y*self.y, self.z*self.z
        wx, wy, wz = self.w*self.x, self.w*self.y, self.w*self.z
        xy, xz, yz = self.x*self.y, self.x*self.z, self.y*self.z

        return np.array([
            [1 - 2*y2 - 2*z2,     2*xy - 2*wz,     2*xz + 2*wy],
            [    2*xy + 2*wz, 1 - 2*x2 - 2*z2,     2*yz - 2*wx],
            [    2*xz - 2*wy,     2*yz + 2*wx, 1 - 2*x2 - 2*y2]
        ])

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


class ViewTransform:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.near = 0.1
        self.far = 1000.0
        self.fov = 60.0  # 视野角度
        self.aspect = width / height
        
    def perspective_matrix(self):
        """
        创建透视投影矩阵
        """
        f = 1.0 / np.tan(np.radians(self.fov) / 2.0)
        return np.array([
            [f/self.aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (self.far+self.near)/(self.near-self.far), 
                  (2*self.far*self.near)/(self.near-self.far)],
            [0, 0, -1, 0]
        ])
        
    def viewport_matrix(self):
        """
        创建视口变换矩阵
        """
    
        return np.array([
            [self.width/2, 0, 0, self.width/2],
            [0, -self.height/2, 0, self.height/2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
def transform_point_to_screen(point, camera, view_transform,flag_filter = True):
    """
    将3D点转换到屏幕空间
    """
    flag_ok = True
    # 转换为齐次坐标
    point_h = np.append(point, 1.0)
    # 1. 视图变换：世界空间 -> 视图空间
    view = np.dot(camera.view_matrix(), point_h)
    # 2. 投影变换：视图空间 -> 裁剪空间
    clip = np.dot(view_transform.perspective_matrix(), view)
    # 3. 透视除法：裁剪空间 -> 标准化设备坐标(NDC)
    ndc = clip / clip[3]
    if clip[3]<=1000:
        flag_ok = False
    if False:#old
        # 4. 视口变换：NDC -> 屏幕空间
        screen = np.dot(view_transform.viewport_matrix(), ndc)
    elif True:
        if True:
            if (ndc[:2]*ndc[:2]).max()>2:
                flag_ok = False
             # 3.1 应用畸变
            cameraDistortion_f = CameraDistortion_f()
            ndc[:2] = cameraDistortion_f.apply_distortion(ndc.reshape(1,4))
            # 4. 视口变换：NDC -> 屏幕空间
            screen = np.dot(view_transform.viewport_matrix(), ndc)
            # print(f'120ndc_old{ndc_old[:2]},ndc,{ndc[:2]},screen_ori:{screen_ori[:2]},screen:{screen[:2]}')

    if flag_ok or not flag_filter:
        return screen[:2]  # 返回屏幕坐标(x,y)
    else:
        return [np.int16(-32768), np.int16(-32768)]

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
    if True:
        screen_coordinates = main()
    elif True:

        camera = QuaternionCamera()
        view_transform = ViewTransform(1280, 960)  # 800x600分辨率



        camera.rotate(yaw, pitch)
        screen_pos = transform_point_to_screen(point, camera, view_transform)

