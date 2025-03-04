import numpy as np
# from jibianjiaozhen import ImprovedCameraPositioner
from fun_jibian import CameraDistortion_f

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
        # return np.array([
        #     [self.width/2, 0, 0, self.width/2],
        #     [0, -self.height/2, 0, self.height/2],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1]
        # ])
    
        return np.array([
            [self.width/2, 0, 0, self.width/2],
            [0, -self.height/2, 0, self.height/2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

class Camera:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 5.0])
        self.target = np.array([0.0, 0.0, 0.0])
        self.up = np.array([0.0, 1.0, 0.0])
        
    def view_matrix(self):
        """
        计算视图矩阵
        """
        # 计算相机坐标系的三个基向量
        z_axis = self.position - self.target  # 前向量
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        x_axis = np.cross(self.up, z_axis)  # 右向量
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)  # 上向量
        
        # 构建视图矩阵
        rotation = np.array([
            [x_axis[0], x_axis[1], x_axis[2], 0],
            [y_axis[0], y_axis[1], y_axis[2], 0],
            [z_axis[0], z_axis[1], z_axis[2], 0],
            [0, 0, 0, 1]
        ])
        
        translation = np.array([
            [1, 0, 0, -self.position[0]],
            [0, 1, 0, -self.position[1]],
            [0, 0, 1, -self.position[2]],
            [0, 0, 0, 1]
        ])
        
        return np.dot(rotation, translation)

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
        if False:
            positioner = ImprovedCameraPositioner()
            # 3.1 应用畸变
            distorted = positioner.distortion.apply_distortion(normalized.reshape(1, 2))
            # 4. 应用相机内参
            pixel_coords = np.dot(positioner.distortion.camera_matrix, 
                                np.append(distorted[0], 1))[:2]
        
        else:

            if (ndc[:2]*ndc[:2]).max()>2:
                flag_ok = False
            # screen_ori = np.dot(view_transform.viewport_matrix(), ndc)
            # ndc_old= ndc.copy()
            
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

# 使用示例
def main():
    # 创建800x600的视口
    view_transform = ViewTransform(800, 600)
    camera = Camera()
    
    # 定义一些3D点
    points = [
        np.array([0.0, 0.0, 0.0]),  # 原点
        np.array([1.0, 1.0, 0.0]),  # 前方一点
        np.array([-1.0, 1.0, 0.0]), # 左上方一点
    ]
    
    # 转换到屏幕空间
    screen_points = []
    for point in points:
        screen_point = transform_point_to_screen(point, camera, view_transform)
        screen_points.append(screen_point)
        print(f"3D点 {point} 在屏幕上的坐标: {screen_point}")
    
    return screen_points

if __name__ == '__main__':
    # 运行示例
    screen_coordinates = main()