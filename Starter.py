"""
Hi all, this is the starting point of the Machine Learning challenge.

Follow the instructions on the pdf and remember to comment your code for 
clarity.

The library for technical indicators is here: 
    
    https://github.com/twopirllc/pandas-ta

The number of indicators should be 84, but it depends how you decide to code
them. As a consequence, remember to vary the size of the input layer in the 
neural network.

Good luck!

Gabriele
"""

import pandas as pd
import pandas_ta as ta

df = pd.read_csv('Downloads/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv', sep=',', header=0)

# Clean NaN values
df = df.dropna(axis=0)

# Initialize Bollinger Bands Indicator
df2=ta.bbands(close=df["Close"], lenght = 20, std = 2)

# Add Bollinger Bands features
df['bbL']=df2.iloc[:,0]
df['bbM']=df2.iloc[:,1]
df['bbU']=df2.iloc[:,2]

# Add Awesome Oscillator
df["ao"]=ta.ao(high=df["High"],low=df["Low"])

# Add more stuff
df["apo"]=ta.apo(close=df["Close"])
df["bop"]=ta.bop(open_=df["Open"],high=df["High"],low=df["Low"],close=df["Close"])
df["cci"]=ta.cci(high=df["High"],low=df["Low"],close=df["Close"])
df["cg"]=ta.cg(close=df["Close"])
df["cmo"]=ta.cmo(close=df["Close"])
df["cpk"]=ta.coppock(close=df["Close"])
#df["fish"]=ta.fisher(high=df["High"],low=df["Low"])
#df["kst"]=ta.kst(close=df["Close"])
#df["macd"]=ta.macd(close=df["Close"])
df["mom"]=ta.mom(close=df["Close"])
#df["ppo"]=ta.ppo(close=df["Close"])
df["roc"]=ta.roc(close=df["Close"])
df["rsi"]=ta.rsi(close=df["Close"])
#df["rvi"]=ta.rvi(open_=df["Open"],high=df["High"],low=df["Low"],close=df["Close"])
df["slope"]=ta.slope(close=df["Close"])
#df["stoch"]=ta.stoch(high=df["High"],low=df["Low"],close=df["Close"])
df["trix"]=ta.trix(close=df["Close"])
df["tsi"]=ta.tsi(close=df["Close"])
df["uo"]=ta.uo(high=df["High"],low=df["Low"],close=df["Close"])
df["willr"]=ta.willr(high=df["High"],low=df["Low"],close=df["Close"])

''' To complete

Technical Analysis Indicators (by Category)

Momentum (21)
Awesome Oscillator: ao
Absolute Price Oscillator: apo
Balance of Power: bop
Commodity Channel Index: cci
Center of Gravity: cg
Chande Momentum Oscillator: cmo
Coppock Curve: coppock
Fisher Transform: fisher
KST Oscillator: kst
Moving Average Convergence Divergence: macd
Momentum: mom
Percentage Price Oscillator: ppo
Rate of Change: roc
Relative Strength Index: rsi
Relative Vigor Index: rvi
Slope: slope
Stochastic Oscillator: stoch
Trix: trix
True strength index: tsi
Ultimate Oscillator: uo
Williams %R: willr
Moving Average Convergence Divergence (MACD)

Overlap (24)
Double Exponential Moving Average: dema
Exponential Moving Average: ema
Fibonacci's Weighted Moving Average: fwma
High-Low Average: hl2
High-Low-Close Average: hlc3
Commonly known as 'Typical Price' in Technical Analysis literature
Hull Exponential Moving Average: hma
Kaufman's Adaptive Moving Average: kama
Ichimoku Kinkō Hyō: ichimoku
Linear Regression: linreg
Midpoint: midpoint
Midprice: midprice
Open-High-Low-Close Average: ohlc4
Pascal's Weighted Moving Average: pwma
William's Moving Average: rma
Simple Moving Average: sma
Sine Weighted Moving Average: sinwma
Symmetric Weighted Moving Average: swma
T3 Moving Average: t3
Triple Exponential Moving Average: tema
Triangular Moving Average: trima
Volume Weighted Average Price: vwap
Volume Weighted Moving Average: vwma
Weighted Moving Average: wma
Zero Lag Moving Average: zlma

Performance (3)

Log Return: log_return
Percent Return: percent_return
Trend Return: trend_return
Percent Return (Cumulative) with Simple Moving Average (SMA)

Statistics (8)
Kurtosis: kurtosis
Mean Absolute Deviation: mad
Median: median
Quantile: quantile
Skew: skew
Standard Deviation: stdev
Variance: variance
Z Score: zscore
Z Score

Trend (11)
Average Directional Movement Index: adx
Archer Moving Averages Trends: amat
Aroon Oscillator: aroon
Decreasing: decreasing
Detrended Price Oscillator: dpo
Increasing: increasing
Linear Decay: linear_decay
Long Run: long_run
Q Stick: qstick
Short Run: short_run
Vortex: vortex
Average Directional Movement Index (ADX)

Utility (1)
Cross: cross

Volatility (8)
Acceleration Bands: accbands
Average True Range: atr
Bollinger Bands: bbands
Donchian Channel: donchian
Keltner Channel: kc
Mass Index: massi
Normalized Average True Range: natr
True Range: true_range
Average True Range (ATR)

Volume (13)
Accumulation/Distribution Index: ad
Accumulation/Distribution Oscillator: adosc
Archer On-Balance Volume: aobv
Chaikin Money Flow: cmf
Elder's Force Index: efi
Ease of Movement: eom
Money Flow Index: mfi
Negative Volume Index: nvi
On-Balance Volume: obv
Positive Volume Index: pvi
Price-Volume: pvol
Price Volume Trend: pvt
Volume Profile: vp
'''