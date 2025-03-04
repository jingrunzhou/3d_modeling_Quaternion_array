
def resolution_conversion(points,wPixel_real,wPixel_referenc,hPixel_real,hPixel_referenc):
    points[:, 0] *= wPixel_real / wPixel_referenc
    points[:, 1] *= hPixel_real / hPixel_referenc
    return points

