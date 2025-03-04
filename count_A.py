import math

class Count():
    def __init__(self,rack_gap,rack_d,Installation_offset=0):
        self.rack_gap,self.rack_d  = rack_gap,rack_d
        self.Installation_offset =  Installation_offset

    def modify(self,difference):# difference
        self.rack_d *= difference
    def __call__(self, *args, **kwargs):
        d = {}
        # for i in range(-9,10):
        for i in range(-9*2, 10*2):
            i/=2
            r = math.atan((i*self.rack_gap - self.Installation_offset) /self.rack_d)
            a = math.degrees(r)
            # a = round(math.degrees(r))
            d[i] = a
            print(f"看第{i}架煤壁,相机需要旋转{a}度")
        
        return d


if __name__ == '__main__':
    # rack_gap = 1.75 #单位：米
    # rack_d = 3.5 #单位：米
    # count = Count(rack_gap,rack_d)
    # print(count())
    # count.modify(0.9)
    # print(count())
    if False:#办公室搭建的环境
        Magnification = 3
        # 矩形和间隔尺寸
        rect_width_mm = 14 * Magnification
        rack_gap = rect_width_mm + 17.4 #单位：mm
        rack_d = 220 #  摄像头距离煤壁距离  单位：mm 

    else:
        rack_gap = 1400 #护帮板中心距 单位：mm
        rack_d = 3220 #  摄像头距离煤壁距离  单位：mm 
        Installation_offset = 200

    count = Count(rack_gap,rack_d,Installation_offset)
    print(count())





