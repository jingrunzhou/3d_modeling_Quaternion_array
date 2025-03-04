import cv2
from Camera_Principles_GD import Camera_Principles
import numpy as np
import pickle  
import os

# 打开视频文件
input_video_path = r'D:\dataset\Virtual_plate_number\solov2\video\7-1731029629037-034.h264'  # 原始视频路径
# input_video_path = r'/data/ins/dataset/Virtual_plate_number/solov2/video/7-1731029629037-034.h264'
output_video_path = input_video_path.split('.')[0] +'_out'+ '.mp4' # 输出视频路径

cap = cv2.VideoCapture(input_video_path)

# 获取视频的宽度、高度和帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print('width,width,fps',width,width,fps)
# 定义输出视频编码格式和输出文件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 设置线的起点和终点坐标
line_start = (50, height // 2)
line_end = (width - 50, height // 2)
# line_color = (0, 255, 0)  # 绿色
line_color = (0, 255, 0)  # 绿色
line_thickness = 4



camera_Principles = Camera_Principles()
camera_Principles.hPixel_real, camera_Principles.wPixel_real = 960,1280
camera_Principles.init_jr()
# np.load('points_plate_fold.npy')
# [points_4d,points_plate_fold,points_line,points_boundary] = np.load('points_plate_fold.npy')


# 从文件中读取对象
with open('points_plate_fold.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# print(loaded_data)
[points_4d,points_plate_fold,points_line,points_boundary] = loaded_data

# 逐帧处理视频
id = 0
while cap.isOpened():
    ret, frame = cap.read()
    id +=1
    if not ret:
        break
    # elif id < 700 or id >1200:
    #     continue

    elif id %1 !=0:
        continue
    print('id',id)

    # 在当前帧上绘制线
    # cv2.line(frame, line_start, line_end, line_color, line_thickness)

    cv2.imwrite(os.path.join(r'D:\dataset\Virtual_plate_number\solov2\video' +'_img', f'IDdhz33_T00_P145_index{id}.jpg'), frame)
    # cv2.imwrite(os.path.join(r'/data/ins/dataset/Virtual_plate_number/solov2/video' +'_img', f'IDdhz33_T00_P145_index{id}.jpg'), frame)



    frame = camera_Principles.single_mark_lines_palte_fold(points_plate_fold, frame)
    # frame = camera_Principles.mark_4point(points_4d, frame)
    # frame = camera_Principles.mark_lines(points_line, frame)
    



    # 写入帧到输出视频
    out.write(frame)

# 释放资源
cap.release()
out.release()



print("视频处理完成！")





