# utils/config.py 这是个基础配置，剩下的请自己创建文件夹进行调用
import os

def is_docker():
    return os.path.exists('/root/biddingTrainEnv')

def get_config():
    return {
        'IS_DOCKER': is_docker(),
        # 其他配置项...
    }

