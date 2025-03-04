import numpy as np
import cv2
import random
from shapely.geometry import Polygon

def xywh_to_points(x, y, w, h):
    """
    将中心点坐标和宽高表示法（xywh）转换为四个角点表示法

    参数：
    - x: 边界框中心点的 x 坐标
    - y: 边界框中心点的 y 坐标
    - w: 边界框的宽度
    - h: 边界框的高度

    返回值：
    - 四个角点的坐标 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    x_min = x - w / 2
    y_min = y - h / 2
    x_max = x + w / 2
    y_max = y + h / 2
    
    point1 = (x_min, y_min)  # 左上角
    point2 = (x_max, y_min)  # 右上角
    point3 = (x_max, y_max)  # 右下角
    point4 = (x_min, y_max)  # 左下角
    points = np.array([point1, point2, point3, point4],dtype=np.int32)
    # print(f"四个角点坐标: {points}")
    return points


def calculate_polygon_area(points):
    # 该函数计算结果有问题
    """
    计算由四个顶点组成的四边形的面积

    参数：
    - points: 四个顶点的坐标 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    返回值：
    - 四边形的面积
    """
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]

    area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) + x4 * (y2 - y1))
    return area


def compute_iou(poly1, poly2):
    """
    计算两个四边形的IoU

    参数：
    - poly1: 第一个四边形的顶点列表 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    - poly2: 第二个四边形的顶点列表 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    返回值：
    - IoU 值
    """
    polygon1 = Polygon(poly1)
    polygon2 = Polygon(poly2)

    if not polygon1.is_valid or not polygon2.is_valid:
        raise ValueError("Invalid polygons provided.")

    # 计算交集多边形
    intersection = polygon1.intersection(polygon2)

    # 计算交集面积和两个多边形的面积
    inter_area = intersection.area
    # poly1_area = polygon1.area
    # poly2_area = polygon2.area

    # 计算并集面积
    # union_area = poly1_area + poly2_area - inter_area

    # 计算IoU
    # iou = inter_area / union_area if union_area != 0 else 0

    area_box = calculate_polygon_area(poly2)


    # iou = inter_area / area_box
    iou = inter_area / polygon2.area


    return iou

def point_in_quadrilateral(point, vertices):
    """
    判断点(px, py)是否在四边形内部

    参数：
    - px, py: 待判断的点的坐标
    - vertices: 四边形的顶点坐标 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    返回值：
    - True 如果点在四边形内部，否则 False
    """
    px, py = point
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    x4, y4 = vertices[3]

    # 计算点到四边形各顶点的向量
    vectors = [
        (px - x1, py - y1),
        (px - x2, py - y2),
        (px - x3, py - y3),
        (px - x4, py - y4)
    ]

    # 计算四边形相邻边的向量
    edges = [
        (x2 - x1, y2 - y1),
        (x3 - x2, y3 - y2),
        (x4 - x3, y4 - y3),
        (x1 - x4, y1 - y4)
    ]

    # 计算叉乘
    cross_products = []
    for i in range(4):
        # 计算点到相邻顶点的向量与相邻边的向量的叉乘
        cross_product = vectors[i][0] * edges[i][1] - vectors[i][1] * edges[i][0]
        cross_products.append(cross_product)

    # 如果所有叉乘结果符号相同，则点在四边形内部
    if all(cp >= 0 for cp in cross_products) or all(cp <= 0 for cp in cross_products):
        result =  True
    else:
        result = False
    print(f"点 {point} 是否在四边形内部？ {result}")
    return result






def Judging_personnel_intrusion(danger_zone,personel_points):
    # boxes = personel_boxes
    # personel_boxes = np.array([[100, 300], [400, 300], [400, 600], [100, 600]])

    flag_test = True
    intrusion_id = []
    if flag_test:
        # 定义图片的宽度和高度
        width = 1000
        height = 1000
        # 创建一个黑色的空白图片
        image = np.zeros((height, width, 3), np.uint8)
        # 绘制危险区域
        cv2.polylines(image, [danger_zone], isClosed=True, color=(255, 0, 0), thickness=2)


    if True:# 检查每个人是否在危险区域内
        for i in range(len(personel_points)):
            points = personel_points[i]
            center = points[2]
            list_points = [points[2],points[3],center]
            flag_intrusion = False
            if False:
                pass

            elif False:#危险区域是四边形，多边形不行
                for list_point in list_points:
                    if point_in_quadrilateral(list_point, danger_zone):
                        # 如果中心点在多边形内部，表示闯入危险区域
                        flag_intrusion = True

            elif False:#危险区域是多边形也可以判断
                for list_point in list_points:
                    list_point = [int(list_point[0]),int(list_point[1])]
                    # 使用 cv2.pointPolygonTest 检测点是否在多边形内
                    if cv2.pointPolygonTest(danger_zone,list_point, False) >= 0:
                        # 如果中心点在多边形内部，表示闯入危险区域
                        flag_intrusion = True

            elif True:# 危险区域是多边形也可以判断
                extend_pix = 5
                point0_ = [points[3][0],max(points[3][1]-extend_pix,0)]
                point1_ = [points[2][0],max(points[2][1]-extend_pix,0)]

                point2_ = [points[2][0],min(points[2][1]+extend_pix,height)]
                point3_ = [points[3][0],min(points[3][1]+extend_pix,height)]

                # cv2.polylines(image, [np.array([point0_,point1_,point2_,point3_],dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

                poly1 = danger_zone
                poly2 = [point0_,point1_,point2_,point3_]
                iou = compute_iou(poly1, poly2)
                print('iou',iou)
                if iou > 0.1 :
                    flag_intrusion = True
                    intrusion_id.append(i)
                else:
                    flag_intrusion = False
            
            if flag_test:
                print('flag_intrusion',flag_intrusion)
                if flag_intrusion:
                    # 如果中心点在多边形内部，表示闯入危险区域
                    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 用红色标记
                    color_ = (0, 0, 255) # 用红色标记
                else:
                    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 用绿色标记
                    color_ =  (0, 255, 0) # 用绿色标记
                cv2.polylines(image, [points], isClosed=True, color=color_, thickness=2)
    cv2.imwrite('./intrusion.jpg',image)
    print('intrusion_id',intrusion_id)
    return intrusion_id
                

def generate_random_yolo_boxes(num_boxes, image_size=(1000, 1000)):
    boxes = []
    for _ in range(num_boxes):
        # 随机生成框的中心点坐标 (x_center, y_center)
        x_center = random.randint(0, image_size[0])
        y_center = random.randint(0, image_size[1])

        # 随机生成框的宽度和高度
        width = random.randint(100, 150)
        height = random.randint(150, 250)

        # 将框的中心点坐标转换为左上角坐标 (x_min, y_min)
        x_min = max(0, x_center - width // 2)
        y_min = max(0, y_center - height // 2)

        # 计算框的 xywh 格式
        box = (x_min, y_min, width, height)
        boxes.append(box)

    return boxes




if __name__ == "__main__":

    
    # 示例：生成100个随机的 YOLO 检测框
    # num_boxes = 100
    # boxes = generate_random_yolo_boxes(num_boxes)

    # # 示例
    # poly1 = [(150, 125), (250, 125), (250, 175), (150, 175)]
    # poly2 = [(200, 150), (300, 150), (300, 200), (200, 200)]

    # iou = compute_iou(poly1, poly2)
    # print(f"IoU: {iou:.4f}")
    # danger_zone = np.array([[400, 200], [600, 180], [700, 800], [100, 600]])

    # danger_zone = np.array([[400, 200], [600, 180], [500, 800], [100, 600],[50, 300],[200, 300]])
    danger_zone = np.array([[100, 200], [500, 280],[500, 600],[800, 600], [900, 700], [500, 800],[300, 800]])

    personel_boxes =  [
    (650, 350, 100, 185),
    (500, 200, 110, 190),
    # (100, 50, 175, 150),
    (300, 200, 110, 220),
    (600, 80, 90, 120),
    (150, 100, 90, 210),
    (580, 680, 115, 180),
    (360, 680, 140, 180),
    (860, 420, 100, 180),
    (560, 490, 150, 180),
]
    # 示例：生成100个随机的 YOLO 检测框
    num_boxes = 50
    personel_boxes = generate_random_yolo_boxes(num_boxes)


    personel_points = []
    for box in personel_boxes:
        x, y, w, h = box
        points = xywh_to_points(x, y, w, h)
        personel_points.append(points)



    intrusion_id = Judging_personnel_intrusion(danger_zone,personel_points)
    # 示例
    # vertices = [(1, 1), (4, 1), (4, 4), (1, 4)]
    # point = (2, 2)
    # result = point_in_quadrilateral(point, vertices)
    # 

    # 如果是底边，不做判断，旋转摄像机，直到不是底边（底边距离下边缘50像素），

    # 如果是底边，则将人员框扩展（按照比例，或者扩展固定像素），将危险区域再做一些扩展，看扩展后还在不在危险区域。





