# rl

key design questions
1. what skill should the agent learn
2. what information does the agent need
3. what actions can the agent take
4. how do we measure success
5. when should episodes end?

what are my thoughts for RL?
i think that we can build a reinforcement learning algorithm
to train a trading agent who takes action in some market
using the information of that market (orderbook)
to find potential entry and exit opportunities based on short term price discrepancies
this can be for pair trading, mean reversion, etc
(i need to find an underlying that would make for a good env and also the data for that env)'


# 4 pm notes


1. created data pipeline and test 2 scripts which gather ohlc from spy and train a minimal rl model based on sharpe
"feature" is any additional piece of information you hand the agent at each timestep to help it decide.
 Common categories beyond raw returns:
Technical: moving average crossovers, RSI, MACD, realized volatility over different windows
Volume-based: volume relative to its own moving average, volume-weighted price
Microstructure: bid-ask spread, order book imbalance (what you're asking about)
Cross-asset: VIX level, sector ETF returns, correlated names

this is where alpha lies

learn patterns within price action
price action 
time price bid ask size volume
after t + 1.