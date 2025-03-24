# main_attention_iql.py

import numpy as np
import torch
import os
import sys
import cProfile
import pstats
import io
import logging
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from run.zjy.zjy_run_attentioniql import run_iql  # 请确保路径正确

torch.manual_seed(1)
np.random.seed(1)

from utils.zjy_config import get_config
from utils.logger import setup_logger
        
def profile_run(cfg, logger):
    run_iql(cfg, logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IQL with custom config")
    parser.add_argument("--config", type=str, help="Path to custom YAML config file")
    args = parser.parse_args()
    
    cfg = get_config(args.config)
    logger = setup_logger(cfg)
    
    if cfg.use_profiler:
        # 使用cProfile运行代码
        pr = cProfile.Profile()
        pr.enable()
        
        profile_run(cfg, logger)
        
        pr.disable()
        
        # 将性能数据保存到文件
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        
        # 将结果写入文件
        with open('profile_results.txt', 'w') as f:
            f.write(s.getvalue())
        
        # 保存为可以用snakeviz查看的格式
        pr.dump_stats('profile_results.prof')

        logger.info("Performance profile saved. You can view it using snakeviz:")
        logger.info("snakeviz profile_results.prof")
    else:
        # 直接运行，不进行性能分析
        run_iql(cfg, logger)