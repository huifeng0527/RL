"""
仿真配置加载模块
"""
import yaml
import os
from pathlib import Path


class SimulationConfigLoader:
    """仿真配置加载器类"""
    
    def __init__(self, config_dir=None):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录，如果为None则使用默认路径
        """
        if config_dir is None:
            # 默认配置文件目录（相对于simulation目录）
            config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
        
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
    
    def get_env_config(self):
        """获取环境配置"""
        return self.load_config('env_config.yaml')
    
    def get_train_config(self):
        """获取训练配置"""
        return self.load_config('train_config.yaml')

