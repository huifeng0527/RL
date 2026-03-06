"""
工作空间处理模块
用于裁剪和处理工作空间图像
"""
import numpy as np


def get_workspace(img):
    """
    获取工作空间区域
    
    Args:
        img: 输入图像
        
    Returns:
        np.ndarray: 裁剪后的工作空间图像
    """
    # 获取图像的高度和宽度
    height, width = img.shape[:2]

    # 定义工作空间为图像中心的一个矩形区域
    # 从左边600像素开始，到右边400像素结束
    workspace = img[:, 600:-400]
    
    return workspace

