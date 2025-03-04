import cv2
import numpy as np

# 旋转向量 (罗德里格斯向量)
rvec = np.array([1.0, 0.5, 0.3])
rvec = np.array([0,0.0, 0])

# 将旋转向量转换为旋转矩阵
rotation_matrix, _ = cv2.Rodrigues(rvec)

print("Rotation Matrix:")
print(rotation_matrix)








