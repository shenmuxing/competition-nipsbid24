from utils.user import *

if USERNAME == "zjy":
    # from .player_bidding_strategy import PlayerBiddingStrategy as PlayerBiddingStrategy
    # from .iql_bidding_strategy import IqlBiddingStrategy as PlayerBiddingStrategy
    # from .bc_bidding_strategy import BcBiddingStrategy as PlayerBiddingStrategy
    # from .onlinelp_bidding_strategy import OnlineLpBiddingStrategy as PlayerBiddingStrategy
    # from .yhq_strategy.yhq_iql_bidding_strategy import IqlBiddingStrategy as PlayerBiddingStrategy
    # from .zjy_strategy.zjy_iql_bidding_strategy import IqlBiddingStrategy as PlayerBiddingStrategy
    from .zjy_strategy.zjy_aiql2_bidding_strategy import IqlBiddingStrategy as PlayerBiddingStrategy

if USERNAME == "yhq":
    # from .player_bidding_strategy import PlayerBiddingStrategy as PlayerBiddingStrategy
    # from .yhq_strategy.yhq_iql_bidding_strategy import IqlBiddingStrategy as PlayerBiddingStrategy
    # from .bc_bidding_strategy import BcBiddingStrategy as PlayerBiddingStrategy
    # from .onlinelp_bidding_strategy import OnlineLpBiddingStrategy as PlayerBiddingStrategy
    from .zjy_strategy.zjy_aiql2_bidding_strategy import IqlBiddingStrategy as PlayerBiddingStrategy