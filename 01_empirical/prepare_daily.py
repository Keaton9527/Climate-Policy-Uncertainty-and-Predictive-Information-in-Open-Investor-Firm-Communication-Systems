"""
生成‘混合特征数据’，将windows生成的含有文本特征的基础数据按照日度进行聚合，再和含有技术因子的 OHLCV 数据合并。

"""

import pandas as pd
from tqdm import tqdm
import warnings
from pathlib import Path

import argparse

# Pandas 3.0 已移除 SettingWithCopyWarning；兼容新旧 Pandas 版本。
setting_with_copy_warning = getattr(pd.errors, 'SettingWithCopyWarning', Warning)
warnings.filterwarnings('ignore', category=setting_with_copy_warning)
warnings.filterwarnings('ignore', category=FutureWarning)

def generate_text_factor(text_url):
    """
    生成文本特征因子

    参数:
    text_url: str, 文本数据文件路径

    返回:
    tuple: (df, drop_codes)
        - df: DataFrame, 包含问题相关特征和回答相关特征的DataFrame
        - drop_codes: list, 需要剔除的股票代码列表(提问数量最少的前5%)
    """
    text_df = pd.read_csv(text_url, converters={'stock_code':str})

    # 拆分问题和答案为两个 dataframe，方便后续按日度聚合
    ques_cols = ['stock_code', 'datetime', 'question', 'question_length', 'q_sentiment_score', 'q_sentiment_label']
    ques_df = text_df[ques_cols].copy()

    ans_cols = ['stock_code', 'whether_answer', 'reply_time', 'reply', 'reply_length', 'a_sentiment_score', 'a_sentiment_label', 'time_diff', 'similarity']
    ans_df = text_df[ans_cols].copy()

    # 重命名，将所有的ques_df 中的列名改成以 ques_ 开头，ans_df 中的列名改成以 ans_ 开头
    ques_df.rename(columns={'q_sentiment_score': 'ques_sentiment_score', 'q_sentiment_label': 'ques_sentiment_label', 'question_length': 'ques_length'}, inplace=True)
    ans_df.rename(columns={'reply_time': 'datetime', 'a_sentiment_score': 'ans_sentiment_score', 'a_sentiment_label': 'ans_sentiment_label', 'reply_length': 'ans_length', 'time_diff': 'ans_time_diff', 'similarity': 'ans_similarity'}, inplace=True)

    # 将时间改为pd.to_datetime 格式
    ques_df['datetime'] = pd.to_datetime(ques_df['datetime'], format='mixed')
    ans_df['datetime'] = pd.to_datetime(ans_df['datetime'], format='mixed')

    # --------------------------------------------------使用2010年1月4日之后的数据--------------------------------------------------
    ques_df = ques_df[ques_df['datetime'] >= '2010-01-04']
    ans_df = ans_df[ans_df['datetime'] >= '2010-01-04']

    # 将15点之后的日期延后一个日历日，随后映射交易日
    ques_df['datetime'] = ques_df['datetime'].apply(lambda x: x + pd.Timedelta(days=1) if x.hour >= 15 else x)
    ans_df['datetime'] = ans_df['datetime'].apply(lambda x: x + pd.Timedelta(days=1) if x.hour >= 15 else x)

    # 将时间改为年月日格式，但是仍然为 datetime64格式
    ques_df['datetime'] = pd.to_datetime(ques_df['datetime'].dt.strftime('%Y-%m-%d'))
    ans_df['datetime'] = pd.to_datetime(ans_df['datetime'].dt.strftime('%Y-%m-%d'))

    # 回答数目
    ques_df['ques_counts'] = 1
    ans_df['ans_counts'] = 1

    # 将回复时间 time_diff 大于2个月（44个日历日）的行按照 whether_answer 为0处理
    # 对whether_answer 为0的行，将time_diff 设置为3个月（66个日历日），回答文本长度（reply_length）记为0，回答的情感极性（a_sentiment_label）为-1，回答的情感强度（a_sentiment_score）记为-1，回答相似度（similarity）记为0；
    ans_df.loc[ans_df['ans_time_diff'] > 44, 'whether_answer'] = 0
    ans_df.loc[ans_df['whether_answer'] == 0, 'ans_length'] = 0
    ans_df.loc[ans_df['whether_answer'] == 0, 'ans_sentiment_label'] = -1
    ans_df.loc[ans_df['whether_answer'] == 0, 'ans_sentiment_score'] = -1
    ans_df.loc[ans_df['whether_answer'] == 0, 'ans_similarity'] = 0
    ans_df.loc[ans_df['whether_answer'] == 0, 'ans_time_diff'] = 66
    ans_df.loc[ans_df['whether_answer'] == 0, 'ans_counts'] = 0

    # 筛选出提问数量最少的前5%的股票代码
    drop_df = ques_df.groupby('stock_code').agg({'ques_counts': 'sum'}).reset_index()
    drop_df = drop_df.sort_values(by='ques_counts', ascending=True)
    drop_df = drop_df.head(int(len(drop_df) * 0.05))
    drop_codes = drop_df['stock_code'].tolist()

    # 初始化聚合字段列表
    agg_fields = {}

    # 情绪极性和情绪强度
    agg_fields['ques_sentiment_score'] = 'mean'
    agg_fields['ans_sentiment_score'] = 'mean'
    agg_fields['ques_sentiment_label'] = 'mean'
    agg_fields['ans_sentiment_label'] = 'mean'

    # 文本长度
    agg_fields['ques_length'] = 'mean'
    agg_fields['ans_length'] = 'mean'

    # 提问字数和回复字数
    agg_fields['ques_length'] = 'mean'
    agg_fields['ans_length'] = 'mean'

    # 提问数量和回复数量
    agg_fields['ques_counts'] = 'sum'
    agg_fields['ans_counts'] = 'sum'

    # 回答时间差和相似度
    agg_fields['ans_time_diff'] = 'mean'
    agg_fields['ans_similarity'] = 'mean'

    # 按日度聚合计算每只股票当日平均情感指数、平均字数、平均相似度、当日问答总数
    ques_agg_fields = {k:v for k,v in agg_fields.items() if k.startswith('ques_')}
    ans_agg_fields = {k:v for k,v in agg_fields.items() if k.startswith('ans_')}

    # 按日度聚合计算每只股票当日平均情感指数、平均字数、平均相似度、当日问答总数
    ques_agg_df = ques_df.groupby(['stock_code', 'datetime']).agg(ques_agg_fields).reset_index()
    ans_agg_df = ans_df.groupby(['stock_code', 'datetime']).agg(ans_agg_fields).reset_index()

    # 将 ques_df 和 ans_df 按日期合并
    df = pd.merge(ques_agg_df, ans_agg_df, how='outer', on=['stock_code', 'datetime'])

    # 填补缺失值
    df['ques_counts'] = df['ques_counts'].fillna(0)
    df['ans_counts'] = df['ans_counts'].fillna(0)
    df['ques_length'] = df['ques_length'].fillna(0)
    df['ans_length'] = df['ans_length'].fillna(0)
    df['ques_sentiment_score'] = df['ques_sentiment_score'].fillna(0)
    df['ans_sentiment_score'] = df['ans_sentiment_score'].fillna(0)
    df['ans_similarity'] = df['ans_similarity'].fillna(0.5)
    df['ans_sentiment_label'] = df['ans_sentiment_label'].fillna(0)
    df['ques_sentiment_label'] = df['ques_sentiment_label'].fillna(0)
    # 当日若没有回复，则回答的时间差为0，回答的情感极性为0，回答的情感强度为0，回答的相似度为0.5
    df['ans_time_diff'] = df['ans_time_diff'].fillna(0)

    return df, drop_codes

def read_ohlcv(ohlcv_url):
    """
    读取OHLCV数据

    参数:
    ohlcv_url: str, OHLCV数据文件路径

    返回:
    df: DataFrame, 包含OHLCV数据的DataFrame
    """
    df = pd.read_csv(ohlcv_url, converters={'stock_code':str})
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d')
    df['stock_code'] = df['stock_code'].apply(lambda x: x[:6])

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aggregate historical record-level text features and align them to market days.")
    parser.add_argument('--interactions', type=Path, required=True)
    parser.add_argument('--market-features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.interactions.resolve(), args.market_features.resolve()}:
        parser.error('Output must differ from input files')
    text_url, ohlcv_url = args.interactions, args.market_features
    print("正在生成文本特征...")
    text_df, drop_codes = generate_text_factor(text_url)
    print("正在读取OHLCV数据...")
    ohlcv_df = read_ohlcv(ohlcv_url)

    # 删除drop_codes 中的股票
    ohlcv_df = ohlcv_df[~ohlcv_df['stock_code'].isin(drop_codes)]

    # 删除text_df 中stock_code 在drop_codes 中的行
    text_df = text_df[~text_df['stock_code'].isin(drop_codes)]

    # The historical IPO-date assignment did not remove observations.
    # The submitted manuscript retains this sample without a 90-day exclusion.

    # 将df 和 ohlcv_df 按stock_code 和 datetime 合并
    df = pd.merge(text_df, ohlcv_df, how='outer', on=['stock_code', 'datetime'])

    # 填补缺失值
    df['ques_counts'] = df['ques_counts'].fillna(0)
    df['ans_counts'] = df['ans_counts'].fillna(0)
    df['ques_length'] = df['ques_length'].fillna(0)
    df['ans_length'] = df['ans_length'].fillna(0)
    df['ques_sentiment_score'] = df['ques_sentiment_score'].fillna(0)
    df['ans_sentiment_score'] = df['ans_sentiment_score'].fillna(0)
    df['ans_similarity'] = df['ans_similarity'].fillna(0.5)
    df['ans_sentiment_label'] = df['ans_sentiment_label'].fillna(0)
    df['ques_sentiment_label'] = df['ques_sentiment_label'].fillna(0)
    # 当日若没有回复，则回答的时间差为0，回答的情感极性为0，回答的情感强度为0，回答的相似度为0.5
    df['ans_time_diff'] = df['ans_time_diff'].fillna(0)

    # ohlcv 是交易日数据，而 text_df 中可能涵盖非交易日的文本信息，因此需要将非交易日的文本信息与接下来第一个交易日的文本因子取均值合并
    # 获取所有交易日

    print("正在处理非交易日数据...")
    # 对每个股票进行处理
    for stock in tqdm(df['stock_code'].unique(), desc="处理股票"):
        trading_dates = sorted(ohlcv_df[ohlcv_df.loc[:, 'stock_code'] == stock]['datetime'].unique())
        stock_data = df[df['stock_code'] == stock].copy()

        # 找出非交易日的数据
        non_trading_dates = stock_data[~stock_data['datetime'].isin(trading_dates)]

        # 按日期排序
        non_trading_dates = non_trading_dates.sort_values('datetime')

        # 在df 中删除该 stock 的非交易日数据
        df = df[~((df['stock_code'] == stock) & (df['datetime'].isin(non_trading_dates['datetime'])))]

        while not non_trading_dates.empty:
            # 获取当前最早的非交易日
            current_date = non_trading_dates['datetime'].iloc[0]

            # 找到下一个交易日
            next_trading_date = next((d for d in trading_dates if d > current_date), None)

            if next_trading_date:
                # 获取当前至下一个交易日的所有非交易日数据
                current_data = stock_data[(stock_data['datetime'] >= current_date) &
                                        (stock_data['datetime'] < next_trading_date)]

                # 获取下一个交易日的数据
                next_data = stock_data[stock_data['datetime'] == next_trading_date]

                if not next_data.empty:
                    # 计算均值
                    text_columns = [col for col in current_data.columns if col.startswith(('ques_', 'ans_'))]
                    for col in text_columns:
                        if col in next_data.columns:
                            # 计算加权平均：当前非交易日数据之和 + 下一个交易日数据
                            next_data[col] = (current_data[col].sum() + next_data[col].values[0]) / (len(current_data) + 1)

                    # 更新数据
                    df.loc[(df['stock_code'] == stock) & (df['datetime'] == next_trading_date), text_columns] = next_data[text_columns].values[0]

                # 从non_trading_dates中移除已处理的数据
                non_trading_dates = non_trading_dates[~non_trading_dates['datetime'].isin(current_data['datetime'])]
            else:
                # 如果没有下一个交易日，直接删除这些数据
                non_trading_dates = non_trading_dates[non_trading_dates['datetime'] > current_date]


    print("正在保存数据...")
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print("数据处理完成！")
