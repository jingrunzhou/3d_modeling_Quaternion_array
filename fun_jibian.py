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
    

