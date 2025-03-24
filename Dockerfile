FROM competition-hub-registry.cn-beijing.cr.aliyuncs.com/alimama-competition/bidding-results:base

# 设置工作目录为/root
WORKDIR /root/biddingTrainEnv

# 将当前目录内容复制到位于/root的容器中
COPY bidding_train_env /root/biddingTrainEnv/bidding_train_env

COPY bidding_train_env/common bidding_train_env/common
# saved_model肯定要
COPY saved_model /root/biddingTrainEnv/saved_model
COPY run /root/biddingTrainEnv/run

COPY utils /root/biddingTrainEnv/utils
COPY config /root/biddingTrainEnv/config
COPY requirements.txt .

# 安装requirements.txt中指定的所有依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 当容器启动时运行run_evaluate.py脚本
CMD ["python3", "./run/run_evaluate.py"]

ENV PYTHONPATH="/root/biddingTrainEnv:${PYTHONPATH}"
