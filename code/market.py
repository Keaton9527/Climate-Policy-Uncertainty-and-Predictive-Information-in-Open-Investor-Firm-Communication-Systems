import pandas as pd
import numpy as np


def calculate_technical_indicators(df):
    """
    计算技术指标
    params:
        df: DataFrame, 包含 OHLCV 数据
    return:
        DataFrame, 包含计算的技术指标
    """
    df = df.sort_values(['stock_code', 'datetime'])  # 按股票代码和时间排序，确保时序计算正确

    # 基础价格变化指标
    df['swing'] = (df['high'] - df['low']) / df['open'] * 100  # 计算日内振幅百分比

    # MACD指标计算
    def calculate_macd(group):
        close = group['close']  # 获取收盘价序列
        exp1 = close.ewm(span=12, adjust=False).mean()  # 计算12日EMA
        exp2 = close.ewm(span=26, adjust=False).mean()  # 计算26日EMA
        diff = exp1 - exp2  # 计算DIFF线，快线减慢线
        dea = diff.ewm(span=9, adjust=False).mean()  # 计算DEA线，DIFF的9日EMA
        macd = 2 * (diff - dea)  # 计算MACD柱，DIFF减DEA乘2
        return pd.DataFrame({'DIFF': diff, 'DEA': dea, 'MACD': macd})  # 返回DataFrame而不是Series

    macd_indicators = df.groupby('stock_code', group_keys=False).apply(calculate_macd)
    df = df.join(macd_indicators)  # 将MACD指标合并到原数据框

    # 移动平均线
    for period in [5, 10, 20, 30, 60]:  # 计算不同周期的移动平均线
        df[f'MA{period}'] = df.groupby('stock_code')['close'].transform(lambda x: x.rolling(window=period).mean())  # 计算period日MA

    # RSI指标计算
    def calculate_rsi(group, periods=14):
        close = group['close']  # 获取收盘价序列
        delta = close.diff()  # 计算价格变化
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()  # 计算上涨平均值
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()  # 计算下跌平均值
        rs = gain / loss  # 计算相对强度
        return 100 - (100 / (1 + rs))  # 计算RSI值

    df['RSI'] = df.groupby('stock_code')['close'].transform(lambda x: calculate_rsi(pd.DataFrame({'close': x})))  # 按股票分组计算RSI

    # CCI指标计算
    def calculate_cci(group, n=14):
        tp = (group['high'] + group['low'] + group['close']) / 3  # 计算典型价格
        ma = tp.rolling(window=n).mean()  # 计算MA
        md = tp.rolling(window=n).apply(lambda x: np.abs(x - x.mean()).mean())  # 计算平均偏差
        cci = (tp - ma) / (0.015 * md)  # 计算CCI值
        return cci

    df['CCI'] = pd.concat([calculate_cci(g) for _, g in df.groupby('stock_code')]).reindex(df.index)  # 按股票分组计算CCI

    # KDJ指标计算
    def calculate_kdj(group, n=9):
        low_min = group['low'].rolling(window=n).min()  # 计算n日最低价
        high_max = group['high'].rolling(window=n).max()  # 计算n日最高价
        rsv = (group['close'] - low_min) / (high_max - low_min) * 100  # 计算未成熟随机值
        k = rsv.ewm(com=2).mean()  # 计算K值
        d = k.ewm(com=2).mean()  # 计算D值
        j = 3 * k - 2 * d  # 计算J值
        return pd.DataFrame({'K': k, 'D': d, 'J': j})  # 返回KDJ三个值

    kdj_indicators = df.groupby('stock_code', group_keys=False).apply(calculate_kdj)
    df = df.join(kdj_indicators)  # 将KDJ指标合并到原数据框

    # ROCP指标计算
    def calculate_rocp(group, n=5):
        return (group['close'] / group['close'].shift(n) - 1) * 100  # 计算n日价格变化率

    df['ROCP'] = df.groupby('stock_code')['close'].transform(lambda x: calculate_rocp(pd.DataFrame({'close': x})))  # 按股票分组计算ROCP

    return df  # 返回添加了技术指标的数据框
