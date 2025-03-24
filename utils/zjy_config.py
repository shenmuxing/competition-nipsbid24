# utils/zjy_config.py

import os
import yaml
from typing import Any, Dict
from utils.user import * 

def is_docker():
    return os.path.exists('/root/biddingTrainEnv')

class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def infer_type(value: str) -> Any:
        """推断并转换值的类型"""
        if not isinstance(value, str):
            return value
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                if value.lower() in ['true', 'yes', 'on', '1']:
                    return True
                elif value.lower() in ['false', 'no', 'off', '0']:
                    return False
                return value

    @classmethod
    def load_yaml(cls, yaml_file):
        with open(yaml_file, 'r',encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 对所有值应用类型推断
        inferred_config = {k: cls.infer_type(v) for k, v in config.items()}
        return cls(inferred_config)

def get_config(custom_yaml_path=None):
    if is_docker():
        base_path = '/root/biddingTrainEnv'
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    config = Config({
        'IS_DOCKER': is_docker(),
        'base_path': base_path,
        'training_data_path': os.path.join(base_path, "data", "traffic","yhq_training_data_rlData_folder"),
        'model_save_path': os.path.join(base_path, "saved_model", f"{USERNAME}_models", "IQLtest"),
        'model_file': os.path.join(base_path, "saved_model", f"{USERNAME}_models", "IQLtest", "iql_model.pth"),
        'normalized_params_path': os.path.join(base_path, "saved_model", f"{USERNAME}_models", "IQLtest", "normalized_params.pkl"),
        'log_interval': 200,
        'num_epochs': 10,
        'use_profiler': False,
        'profile_batch_limit': 50000,
        'use_normalization': True,
        'gamma': 0.99,
        'tau': 0.01,
        'D_curr': 17,
        'D_hist': 14,
        'D_other': 7,
        'network_random_seed': 1,
        'expectile': 0.7,
        'temperature': 3.0,
        'hidden_dim': 64,
        'critic_num_hidden_layers': 2,
        'actor_num_hidden_layers': 2,
        'early_stopping_patience': 2,
        'save_interval': 1,
        'log_file': os.path.join(base_path, "logs", "zjy", "iql_training.log"),
        'log_level': "INFO",
        'log_max_size': 10 * 1024 * 1024,
        'log_backup_count': 5,
        "use_reward_based_sampler": True,
        'log_backup_count': 5,
        "use_reward_based_sampler": True,
        'reward_threshold': 0,
        'alpha': 0.8,
        'beta': 0.1,
        'max_grad_norm': 1.0,
        "memmap_data_path": "memmap_data",

        'iql_num_epochs': 1,
        'iql_batch_size':128,
        'iql_patience': 5,
        'V_lr': 1e-4,
        'critic_lr': 1e-4,
        'actor_lr': 1e-4,
        'iql_save_path': os.path.join(base_path, "saved_model", f"{USERNAME}_models", "IQLtest"),

        'attention_d_model': 72,
        'attention_n_heads': 8,
        'attention_dropout': 0.1,
        'attention_lr': 5e-5,
        'attention_num_epochs': 1,
        'attention_batch_size': 1,
        'attention_patience': 3,
        'attention_save_path': os.path.join(base_path, "saved_model", f"{USERNAME}_models", "AttentionIQL"),
        'attention_train_test_split': 0.9,
        'test_set_train_epoches':1,

        'use_cuda': False,
        'num_workers':0,
    })


    # 如果存在默认的 YAML 配置文件，则使用它覆盖默认配置
    default_yaml_file = os.path.join(base_path, 'config/zjy_default.yaml')
    if os.path.exists(default_yaml_file):
        yaml_config = Config.load_yaml(default_yaml_file)
        config.update(yaml_config)

    # 如果提供了自定义 YAML 配置文件，则使用它进一步覆盖配置
    if custom_yaml_path and os.path.exists(custom_yaml_path):
        custom_yaml_config = Config.load_yaml(custom_yaml_path)
        config.update(custom_yaml_config)

    return config

