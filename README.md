# Competition-NIPS-Bid24

这是埃雅仁迪尔团队参加NIPS2024的代码仓库。比赛的题目是General Track: Auto-Bidding in Uncertain Environment。

Aiya Eärendil Elenion Ancalima!

## Timeline and Rules

- Registration Stage : June 25, 2024 - August 8, 2024
- Practice Round： July 22, 2024 - August 8, 2024
- Official Round
  - Preliminary Round： August 9, 2024 - September 12  All participants are consistently ranked according to their evaluation scores on the leaderboard, with a maximum of 100 teams with the highest scores advancing to the finals.所有参赛者参加，前100名晋级
  - Final Round: September 16, 2024 - October 24, 2024 Advancing participants will get more submission times per day and need to further optimize their agents. Final rankings on the leaderboard will be determined using the same evaluation metrics and methods as in the preliminary round.
  - Announcement of Winning Teams: November, 2024
  - Award Ceremony： December,2024

All deadlines are at 11:59 PM AOE on the corresponding day. The organizers reserve the right to update the contest timeline if necessary.

Top 10 teams should submit a technical report that includes a summary of their solution (suggested length 4 pages), as well as the code for reproducing the results. The technical report needs to include solutions to uncertainty, that is, how to deal with variance in conversion predictions, data sparsity, multi-slots, etc. A technical report that does not include solutions to these problems will be considered invalid.

Only valid submission will be considered any awards and will be put in the final leaderboard of the competition.

## Task

### Problem

The task for this competition is based on a simplified version of Target CPA and defines CPA as the average cost per conversion. Participants are required to design and implement an auto-bidding strategy. Given the advertiser j's budget B and the desired CPA C, the auto-bidding strategy bids for N impression opportunities within an advertising delivery period, the objective is to maximize the total conversion volume while ensuring that the realized CPA remains below C at the end of the period.

本次比赛的任务基于目标CPA的简化版本，将CPA定义为每次转化的平均成本。参赛者需要设计并实现一个自动竞价策略。给定广告主j的预算B和期望CPA C，自动竞价策略在广告投放期内对N个展示机会进行竞价，目标是最大化总转化量，同时确保在期末实现的CPA保持在C以下。

Specifically, all impression opportunities arrive sequentially, and the bidding strategy bids on each opportunity in turn. For each opportunity i:

具体而言，所有展示机会按顺序到达，竞价策略依次对每个机会进行竞价。对于每个机会i：

1. Action Prediction: The conversion action probability when advertiser j's ad is exposed to the customer is predicted as  $pvalue_{i}$(For simplicity, we usually omit the index j of the advertiser where it does not cause ambiguity.), and the variance of prediction represents $\sigma_i$.

   行为预测：当广告主j的广告展示给客户时，转化行为概率被预测为$pvalue_{i}$(这里一般忽略index j 因为这里不会产生模糊？？？？？)，预测的方差表示为$\sigma_i$。
2. Bidding: The auto-bidding strategy bids $b_i$, while other competing advertisers simultaneously bid $b_i^-$ using their individual bidding strategies to compete for 3 ad slots in the opportunity i.

   竞价：自动竞价策略出价$b_i$，而其他竞争广告主同时使用各自的竞价策略出价$b_i^-$，以竞争机会i中的3个广告位。
3. Auction: The ad platform runs a GSP (Generalized Second Price) auction mechanism and returns the auction results to the advertiser j. $x_i(b_i, b_i^-)$ indicates whether to win, $k_i(b_i, b_i^-)$ denotes the ad slot won, and $c_i(b_i, b_i^-)$ represents the cost. $x_i, k_i, c_i$ depends not only on $b_i$ but also on $b_i^-$.

   拍卖：广告平台运行GSP（广义第二价格）拍卖机制，并将拍卖结果返回给广告主j。$x_i(b_i, b_i^-)$表示是否获胜，$k_i(b_i, b_i^-)$表示赢得的广告位，$c_i(b_i, b_i^-)$表示成本。$x_i, k_i, c_i$不仅取决于$b_i$，还取决于$b_i^-$。
4. Impression: Whether the $k_i$-th slot is exposed to the customer is determined by a random variable defined as $E_i$ ~ $\text{Bernoulli}(exposure_{k_i})$, where $exposure_{k_i}$ is the probability of exposure for the slot $k_i$. The actual conversion occurrence is also a random variable defined as $V_i ~ \text{Bernoulli}(value_i)$ where $value_i$ ~ $N(pvalue_i, \sigma_i^2)$, and $\sigma_i$ represents the variance of prediction. If an ad in the slot is not exposed, the advertiser doesn't need to pay the cost and the customer won't make conversion on the ad.

   展示：第$k_i$个广告位是否展示给客户由随机变量$E_i$ ~ $\text{Bernoulli}(exposure_{k_i})$决定，其中$exposure_{k_i}$是广告位$k_i$的展示概率。实际转化发生也是一个随机变量，定义为$V_i$ ~ $\text{Bernoulli}(value_i)$，其中$value_i$ ~ $N(pvalue_i, σ_i^2)$，$σ_i$表示预测的方差。如果广告位中的广告未被展示，广告主无需支付费用，客户也不会对该广告进行转化。

Therefore, this task can be formalized as follows:

因此，这个任务可以形式化如下：

$$
\begin{aligned} \max _{b_{1}, \cdots, b_{N}} & \sum_{i} x_{i}\left(b_{i}, b_{i}^{-}\right) \cdot E_{i} \cdot V_{i} \\ \text { s.t. } & \sum_{i} x_{i}\left(b_{i}, b_{i}^{-}\right) \cdot E_{i} \cdot c_{i}\left(b_{i}, b_{i}^{-}\right) \leq B, \\ & \frac{\sum_{i} x_{i}\left(b_{i}, b_{i}^{-}\right) \cdot E_{i} \cdot c_{i}\left(b_{i}, b_{i}^{-}\right)}{\sum_{i} x_{i}\left(b_{i}, b_{i}^{-}\right) \cdot E_{i} \cdot V_{i}} \leq C .\end{aligned}
$$

The realized CPA of the advertiser $ j $ is $ C P A=\frac{\sum_{i} x_{i}\left(b_{i}, b_{i}^{-}\right) \cdot E_{i} \cdot c_{i}\left(b_{i}, b_{i}^{-}\right)}{\sum_{i} x_{i}\left(b_{i}, b_{i}^-\right) \cdot E_{i} \cdot V_{i}} $.

广告主j的实现CPA为$ C P A=\frac{\sum_{i} x_{i}\left(b_{i}, b_{i}^{-}\right) \cdot E_{i} \cdot c_{i}\left(b_{i}, b_{i}^{-}\right)}{\sum_{i} x_{i}\left(b_{i}, b_{i}^-\right) \cdot E_{i} \cdot V_{i}} $。

### Evaluation Metric

The auto-bidding strategy aims to maximize conversion volume while satisfying the CPA constraint set by the advertiser. If the realized CPA exceeds the advertiser's desired CPA C, a penalty will be applied. The detailed evaluation metrics are defined as follows:

自动竞价策略旨在最大化转化量，同时满足广告主设定的CPA约束。如果实现的CPA超过广告主期望的CPA C，将会应用惩罚。详细的评估指标定义如下：

$$
Score  =\mathbb{P}(C P A ; C) \cdot \sum_{i} x_{i} \cdot E_{i} \cdot V_{i}
$$

In real-world scenarios, if the advertiser's total costs exceeds the budget during the advertising delivery period, the ad platform typically stops the advertiser's bidding for the remaining opportunities. Therefore, the budget constraint can always be satisfied. The current evaluation metric focuses solely on the CPA constraint, where the penalty function $ \mathbb{P}(C P A ; C) $ for exceeding the CPA constraint is defined as:

在实际场景中，如果广告主的总成本在广告投放期间超过预算，广告平台通常会停止广告主对剩余机会的竞价。因此，预算约束始终可以满足。当前的评估指标仅关注CPA约束，超过CPA约束的惩罚函数$ \mathbb{P}(C P A ; C) $定义为：

$$
\mathbb{P}(C P A ; C)=\min \left\{\left(\frac{C}{C P A}\right)^{\beta}, 1\right\}
$$

The penalty function is defined with a hyperparameter $ \beta>0 $，typically set to $ 3 \cdot \mathbb{P}(C P A ; C) $ implies that the penalty is incurred only when $ C P A> $ $ C $.

惩罚函数定义了一个超参数$ \beta>0 $，通常设置为3。$ \mathbb{P}(C P A ; C) $意味着只有当CPA > C时才会产生惩罚。

In each evaluation, the participant's strategy being evaluated is required to bidding on behalf of one designated advertiser, given a specific budget, CPA, and other settings. To fully evaluate the performance of the strategy across various advertisers, we will run it multiple times in the auction system using different advertiser profiles and delivery periods, and then average the results as the evaluation score.

在每次评估中，被评估的参赛者策略需要代表一个指定的广告主进行竞价，给定特定的预算、CPA和其他设置。为了全面评估该策略在各种广告主中的表现，我们将在拍卖系统中使用不同的广告主配置和投放期多次运行它，然后将结果平均作为评估分数。

### Auction System

To comprehensively demonstrate real-world scenarios and ensure the quality of the competition, we have developed a standardized ad auction system specifically for the competition platform.

为了全面展示真实世界的场景并确保比赛的质量，我们专门为竞赛平台开发了一个标准化的广告拍卖系统。

![alt text](images/1jnmz6sl.bmp)

This system effectively reproduces dynamic competition among advertisers, impression opportunity arrival patterns, and several essential industry characteristics. To simplify the auto bidding process for participants, we divide impression opportunities in an advertising delivery period into 48 decision time steps. Given the objective, the auto-bidding strategy sequentially bids for each step, storing its own results from step t in preceding historical information to refine strategies for step t+1. Within each step, all impression opportunities are executed independently and in parallel. At the end of the period, the system provides final performance, including the total conversion volume, whether the realized CPA meets the advertiser's desired CPA, and additional relevant information. The details are as follows:

该系统有效地再现了广告主之间的动态竞争、展示机会到达模式以及几个重要的行业特征。为了简化参赛者的自动竞价过程，我们将广告投放期内的展示机会分为48个决策时间步骤。给定目标，自动竞价策略依次为每个步骤进行竞价，将步骤t的结果存储在先前的历史信息中，以优化步骤t+1的策略。在每个步骤内，所有展示机会都独立并行执行。在期末，系统提供最终表现，包括总转化量、实现的CPA是否满足广告主期望的CPA，以及其他相关信息。详细内容如下：

1. Impression Opportunity: Different customers exhibit unique impression opportunity arrival patterns based on their interests and shopping preferences. For example, those purchasing office supplies typically arrive during daytime hours, while those buying clothing are more inclined to arrive at night. To reflect this reality, we chose representative preferences from real-world industrial scenarios and sampled their traffic arrival patterns to serve as impression opportunities for the auction system.

   展示机会：不同的客户根据其兴趣和购物偏好展现独特的展示机会到达模式。例如，购买办公用品的客户通常在白天到达，而购买服装的客户更倾向于在晚上到达。为了反映这一现实，我们从实际工业场景中选择了具有代表性的偏好，并对其流量到达模式进行采样，作为拍卖系统的展示机会。
2. Agents: Advertisers represent diverse industry categories, with varying budgets and CPAs. Each advertiser uses unique bidding strategies. Competitors' bidding strategies, supplied by the organizer, are trained using multiple methods. These strategies draw from actual industry practices and exhibit varying degrees of effectiveness. Advertisers exhibit different marketing preferences towards different customers. Consequently, for an opportunity, certain advertisers may opt for higher bids, while others might choose lower ones.

   代理：广告主代表不同的行业类别，具有不同的预算和CPA。每个广告主使用独特的竞价策略。竞争对手的竞价策略由组织者提供，使用多种方法进行训练。这些策略源自实际行业实践，并展现不同程度的有效性。广告主对不同客户展现不同的营销偏好。因此，对于一个机会，某些广告主可能选择更高的出价，而其他人可能选择更低的出价。
3. Ad Auction: Each impression opportunity comprises 3 ad slots which will be sequentially exposed from top to bottom to customers. The auction mechanism adopts the GSP (Generalized Second-Price Auction) mechanism. Advertisers are ranked based on their bids, with the top 3 winners allocated to the corresponding 3 ad slots. We charge the advertisers once they have been exposed, so that the payment of winning advertisers are equal to the bid of the next ad slot advertiser.

   广告拍卖：每个展示机会包含3个广告位，将从上到下依次向客户展示。拍卖机制采用GSP（广义第二价格拍卖）机制。广告主根据其出价进行排名，前3名获胜者分配到相应的3个广告位。我们在广告被展示后才向广告主收费，因此获胜广告主的支付等于下一个广告位广告主的出价。

### InterfaceIn this competition, participants can freely use various tools and methods to optimize their auto bidding strategies.

在本次比赛中，参赛者可以自由使用各种工具和方法来优化他们的自动竞价策略。

We have defined standard input and output interfaces for the participants. Therefore, participants only need to complete the implementation of the 'bidding' function in the 'PlayerBiddingStrategy' class, which takes input information and returns the bids for impression opportunities at each decision step.

我们为参赛者定义了标准的输入和输出接口。因此，参赛者只需要完成'PlayerBiddingStrategy'类中'bidding'函数的实现，该函数接收输入信息并返回每个决策步骤的展示机会的出价。

```python
class PlayerBiddingStrategy(BaseBiddingStrategy):
    """
    Standardized interface for Participants.
    """

    def __init__(self, name="PlayerStrategy", budget, cpa, category):
        """
        Initialize the bidding strategy.
        parameters:
            @budget: the advertiser's budget for a delivery period.
            @cpa: the CPA constraint of the advertiser.
            @category: the index of advertiser's industry category.
        """
        super().__init__(name, budget, cpa, category)

    def reset(self):
        """
        Reset the remaining budget to its initial state.
        """
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid, historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        """
        Bids for all the opportunities in a delivery period
  
        parameters:
         @timeStepIndex: the index of the current decision time step. 
         @pValues: the conversion action probability.
         @pValueSigmas: the prediction probability uncertainty.
         @historyPValueInfo: the history predicted value and uncertainty for each opportunity.
         @historyBid: the advertiser's history bid for each opportunity.
         @historyAuctionResult: the history auction result for each opportunity.
         @historyImpressionResult: the history impression result for each opportunity.
         @historyLeastWinningCost: the history least wining cost for each opportunity.
  
        return:
            Return the bids for all the opportunities in the delivery period.
        """


        self.model = self.load_model(...)  # Load the prediction model
        alpha = self.model.predict(...)  # Predict the bid coefficient
        bids = alpha * pv_values  # Calculate the final bid
        return bids
```

The auto-bidding strategy sequentially bids for each time step, utilizing its previous information from step t stored in historical data to refine its strategy for step t + 1. The information contained is as follows:

自动竞价策略依次为每个时间步骤进行竞价，利用存储在历史数据中的步骤t的先前信息来优化步骤t + 1的策略。包含的信息如下：

- historyPValueInfo represents the history predicted value and uncertainty for each opportunity across multiple steps.

  historyPValueInfo表示多个步骤中每个机会的历史预测值和不确定性。
- historyBid represents the advertiser's history bids for each opportunity across multiple steps.

  historyBid表示广告主在多个步骤中对每个机会的历史出价。
- historyAuctionResult represents the history auction results for each opportunity including the winning status, slot, and cost across multiple steps.

  historyAuctionResult表示多个步骤中每个机会的历史拍卖结果，包括获胜状态、广告位和成本。
- historyImpressionResult represents the history impression result for each opportunity, including exposure status and conversion action across multiple steps.

  historyImpressionResult表示多个步骤中每个机会的历史展示结果，包括展示状态和转化行为。
- historyLeastWinningCost represents the history least winning costs for each opportunity across multiple steps.

  historyLeastWinningCost表示多个步骤中每个机会的历史最低获胜成本。

## Dataset

The training dataset is derived from advertising delivery data generated via the auction system where multiple advertisers compete against each other.
训练数据集来源于多个广告主相互竞争的拍卖系统生成的广告投放数据。

Participants can use this dataset to recreate the historical delivery process of all advertisers across all impression opportunities.
参与者可以使用此数据集重现所有广告主在所有展示机会中的历史投放过程。

The training dataset includes many delivery periods. Each delivery period contains approximately 500,000 impression opportunities and is divided into 48 steps.
训练数据集包含多个投放周期。每个投放周期包含约500,000个展示机会，并被分为48个步骤。

There are 48 advertisers competing for these opportunities.
有48个广告主在竞争这些机会。

The specific data format is as follows:
具体的数据格式如下：

- deliveryPeriodIndex: Represents the index of the current delivery period.表示当前投放周期的索引。
- advertiserNumber: Represents the unique identifier of the advertiser.表示广告主的唯一标识符。
- advertiserCategoryIndex: Represents the index of the advertiser's industry category.表示广告主所属行业类别的索引。
- budget: Represents the advertiser's budget for a delivery period.表示广告主在一个投放周期内的预算。
- CPAConstraint: Represents the CPA constraint of the advertiser.表示广告主的CPA约束。"Cost Per Action"（将CPA定义为每次转化的平均成本），具体应该是指他们愿意为每次用户行动（如购买、注册或下载等）支付的最高成本
- timeStepIndex: Represents the index of the current decision time step.表示当前决策时间步骤的索引。
- remainingBudget: Represents the advertiser's remaining budget before the current step.表示当前步骤之前广告主的剩余预算。
- pvIndex: Represents the index of the impression opportunity.表示展示机会的索引。
- pValue: Represents the conversion action probability when the advertisement is exposed to the customer.表示广告展示给客户时的转化行为概率。
- pValueSigma: Represents the prediction probability uncertainty.表示预测概率的不确定性。
- bid: Represents the advertiser's bid for the impression opportunity.表示广告主对展示机会的出价。
- xi: Represents the winning status of the advertiser for the impression opportunity, where 1 implies winning the opportunity and 0 suggests not winning the opportunity.表示广告主是否赢得展示机会，1表示赢得机会，0表示未赢得机会。
- adSlot: Represents the won ad slot. The value ranges from 1 to 3, with 0 indicating not winning the opportunity.表示赢得的广告位。值范围从1到3，0表示未赢得机会。
- cost: Represents the cost that the advertiser needs to pay if the ad is exposed to the customer.表示如果广告展示给客户，广告主需要支付的成本。
- isExposed: Represents whether the ad in the slot was displayed to the customer, where 1 implies the ad is exposed and 0 suggests not exposed.表示广告位中的广告是否展示给客户，1表示广告被展示，0表示未被展示。
- conversionAction: Represents whether the conversion action has occurred, where 1 implies the occurrence of the conversion action and 0 suggests that it has not occurred.表示是否发生了转化行为，1表示发生了转化行为，0表示未发生转化行为。
- leastWinningCost: Represents the minimum cost to win the impression opportunity, i.e., the 4-th highest bid of the impression opportunity.表示赢得展示机会的最低成本，即展示机会的第四高出价。
- isEnd: Represents the completion status of the advertising period, where 1 implies either the final decision step of the delivery period or the advertiser's remaining budget falling below the system-set minimum remaining budget.表示广告投放周期的完成状态，1表示要么是投放周期的最后一个决策步骤，要么是广告主的剩余预算低于系统设定的最低剩余预算。

## 优化方法

1. 强化学习优化对偶问题
2. 强化学习优化原始问题
3. 在线优化方法优化对偶问题：具体地，使用带约束的镜梯度类下降方法[Mirror GD](https://www2.isye.gatech.edu/~nemirovs/LMCOLN2023Spring.pdf).

## 文件树

```
tmp/ # 不会同步这个文件夹，可以在里面做任何事情，包括撸管
tmp.py # 临时文件，可以随时删除,也不会同步
data/ 
 |── traffic 
     |── period-7.csv(feather)
     |── period-8.csv(feather)
     |── period-9.csv(feather)
     |── period-10.csv(feather)
     |── period-11.csv(feather)
     |── period-12.csv(feather)
     |── period-13.csv(feather)
utils/
 |── config.py # 全局配置，其中有判断目前是否在docker环境下的函数
```

其中`data`文件夹如果在本地存储，则使用feather作为数据文件的存储方式，

## Dependencies

这个是原版的安装方法
```
conda create -n nips-bidding-env python=3.9.12 pip=23.0.1
conda activate nips-bidding-env
pip install -r requirements.txt
```

如果cuda版本大于11，以上方法应该不work,下面的是笔者的安装方法
```
conda create -n [environment name] python=3.9.12 pip=23.0.1
conda activate [environment name]
conda install -c conda-forge numpy=1.24.2 pandas=2.0.3 matplotlib=3.3.4 scipy=1.11.1 psutil -y
conda install pytorch=1.12.0 torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge -y
pip install gin==0.1.6 gin-config==0.5.0 func_timeout
pip install pyarrow==17.0.0
```


## 运行

### data preprocessing

Run this script to convert the traffic granularity data into trajectory data required for RL training, facilitating subsequent RL policy training.
别使用原始版本的python，会报错！
```bash
python -m bidding_train_env.dataloader.rl_data_generator
```

### strategy training
#### reinforcement learning-based bidding

##### IQL(Implicit Q-learning) Model

Load the training data and train the IQL bidding strategy.
```
python main/main_iql.py 
```
Use the IqlBiddingStrategy as the PlayerBiddingStrategy for evaluation.
```
bidding_train_env/strategy/__init__.py
from .iql_bidding_strategy import IqlBiddingStrategy as PlayerBiddingStrategy
```
##### BC(behavior cloning) Model

Load the training data and train the BC bidding strategy.
```
python main/main_bc.py 
```
Use the BcBiddingStrategy as the PlayerBiddingStrategy for evaluation.
```
bidding_train_env/strategy/__init__.py
from .bc_bidding_strategy import BcBiddingStrategy as PlayerBiddingStrategy
```

#### online linear programming-based bidding
##### OnlineLp Model

Load the training data and train the OnlineLp bidding strategy.
```
python main/main_onlineLp.py 
```
Use the OnlineLpBiddingStrategy as the PlayerBiddingStrategy for evaluation.
```
bidding_train_env/strategy/__init__.py
# from .onlinelp_bidding_strategy import OnlineLpBiddingStrategy as PlayerBiddingStrategy
```

### offline evaluation

Load the training data to construct an offline evaluation environment for assessing the bidding strategy offline.
```
python main/main_test.py
```
