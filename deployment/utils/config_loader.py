"""
配置加载模块
用于加载和管理YAML配置文件
"""
import yaml
import os
from pathlib import Path


class ConfigLoader:
    """配置加载器类"""
    
    def __init__(self, config_dir=None):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录，如果为None则使用默认路径
        """
        if config_dir is None:
            # 默认配置文件目录（相对于deployment目录）
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            config_dir = os.path.join(base_dir, 'deployment', 'configs')
        
        self.config_dir = Path(config_dir)
        self._configs = {}
    
    def load_config(self, config_name):
        """
        加载指定配置文件
        
        Args:
            config_name: 配置文件名（不含路径）
            
        Returns:
            dict: 配置字典
        """
        if config_name in self._configs:
            return self._configs[config_name]
        
        config_path = self.config_dir / config_name
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self._configs[config_name] = config
        return config
    
    def get_robot_config(self):
        """获取机器人配置"""
        return self.load_config('robot_config.yaml')
    
    def get_camera_config(self):
        """获取摄像头配置"""
        return self.load_config('camera_config.yaml')
    
    def get_model_config(self):
        """获取模型配置"""
        return self.load_config('model_config.yaml')
    
    def get_hand_detection_config(self):
        """获取手部检测配置"""
        return self.load_config('hand_detection_config.yaml')
    
    def resolve_path(self, path, base_dir=None):
        """
        解析相对路径为绝对路径
        
        Args:
            path: 相对路径
            base_dir: 基础目录，如果为None则使用deployment目录
            
        Returns:
            str: 绝对路径
        """
        if os.path.isabs(path):
            return path
        
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            base_dir = os.path.join(base_dir, 'deployment')
        
        return os.path.abspath(os.path.join(base_dir, path))

