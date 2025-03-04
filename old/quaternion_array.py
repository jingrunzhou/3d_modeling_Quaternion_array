import numpy as np
from math import sin, cos, sqrt, pi

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
        self.position = np.array([0.0, 0.0, 0.0])
        self.orientation = Quaternion()  # 初始朝向
        self.right = np.array([1.0, 0.0, 0.0])
        self.up = np.array([0.0, 1.0, 0.0])
        self.forward = np.array([0.0, 0.0, -1.0])

    def rotate(self, yaw, pitch):
        """使用四元数进行相机旋转"""
        # 创建偏航旋转（绕Y轴）
        yaw_rotation = Quaternion.from_axis_angle(np.array([0, 1, 0]), yaw * pi / 180)
        
        # 创建俯仰旋转（绕右向量）
        pitch_rotation = Quaternion.from_axis_angle(self.right, pitch * pi / 180)
        
        # 组合旋转
        self.orientation = yaw_rotation.multiply(pitch_rotation.multiply(self.orientation))
        self.orientation.normalize()
        
        # 更新相机方向向量
        rotation_matrix = self.orientation.to_matrix()
        self.forward = np.dot(rotation_matrix, np.array([0, 0, -1]))
        self.right = np.dot(rotation_matrix, np.array([1, 0, 0]))
        self.up = np.dot(rotation_matrix, np.array([0, 1, 0]))

    def get_view_matrix(self):
        """获取视图矩阵"""
        rotation_matrix = self.orientation.to_matrix()
        view_matrix = np.eye(4)
        
        # 设置旋转部分
        view_matrix[:3, :3] = rotation_matrix.T
        
        # 设置平移部分
        view_matrix[:3, 3] = -np.dot(rotation_matrix.T, self.position)
        
        return view_matrix

    def slerp(self, target_orientation, t):
        """球面线性插值，实现平滑旋转"""
        # 计算四元数点积
        dot = (self.orientation.w * target_orientation.w + 
               self.orientation.x * target_orientation.x +
               self.orientation.y * target_orientation.y + 
               self.orientation.z * target_orientation.z)
        
        # 确保走最短路径
        if dot < 0:
            target_orientation = Quaternion(-target_orientation.w,
                                         -target_orientation.x,
                                         -target_orientation.y,
                                         -target_orientation.z)
            dot = -dot
        
        # 防止除零错误
        if dot > 0.9995:
            return self.orientation
        
        # 计算插值角度
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        
        s0 = cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return Quaternion(
            s0 * self.orientation.w + s1 * target_orientation.w,
            s0 * self.orientation.x + s1 * target_orientation.x,
            s0 * self.orientation.y + s1 * target_orientation.y,
            s0 * self.orientation.z + s1 * target_orientation.z
        )

# 使用示例
def transform_point(point, camera):
    """将世界空间点转换到相机空间"""
    view_matrix = camera.get_view_matrix()
    point_h = np.append(point, 1.0)  # 转换为齐次坐标
    transformed = np.dot(view_matrix, point_h)
    return transformed[:3]

# 创建相机并进行测试
camera = QuaternionCamera()


# 旋转相机
camera.rotate(45, 45)  # 偏航30度，俯仰15度


# 测试点变换
test_point = np.array([1.0, 1.0, 1.0])
transformed_point = transform_point(test_point, camera)
print(transformed_point)
s = 0

for t in transformed_point:
    s+= t**2
print(s)