# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:21:13 2026

@author: user
"""

# -*- coding: utf-8 -*-

## 金融資料視覺化看板
## 最終版:
## 1. 股票數量單位 = 張
## 2. 期貨數量單位 = 口
## 3. 所有 TWD 換算全部在 order_streamlit.py 內部完成


# 載入必要模組
import os
import numpy as np
from talib.abstract import SMA, EMA, WMA, RSI, BBANDS, MACD, STOCH
import itertools, anthropic
import indicator_f_Lo2_short, datetime, indicator_forKBar_short
import pandas as pd
import streamlit as st
import streamlit.components.v1 as stc
from order_streamlit import Record
import matplotlib.pyplot as plt
import matplotlib

import plotly.graph_objects as go
from plotly.subplots import make_subplots


#%%
####### (1) 開始設定 #######
html_temp = """
        <div style="background-color:#3872fb;padding:10px;border-radius:10px">   
        <h1 style="color:white;text-align:center;">金融看板與程式交易平台 </h1>
        <h2 style="color:white;text-align:center;">Financial Dashboard and Program Trading </h2>
        </div>
        """
stc.html(html_temp)


@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def load_data(path):
    df = pd.read_pickle(path)
    return df


st.subheader("選擇金融商品: ")
choices = [
    '台積電: 2022.1.1 至 2024.4.9',
    '大台指期貨2024.12到期: 2023.12 至 2024.4.11',
    '小台指期貨2024.12到期: 2023.12 至 2024.4.11',
    '英業達2020.1.2 至 2024.4.12',
    '堤維西2020.1.2 至 2024.4.12'
]
choice = st.selectbox('選擇金融商品', choices, index=0)

if choice == '台積電: 2022.1.1 至 2024.4.9':
    df_original = load_data('kbars_2330_2022-01-01-2024-04-09.pkl')
    product_name = '台積電2330'

if choice == '大台指期貨2024.12到期: 2023.12 至 2024.4.11':
    df_original = load_data('kbars_TXF202412_2023-12-21-2024-04-11.pkl')
    product_name = '大台指期貨'

if choice == '小台指期貨2024.12到期: 2023.12 至 2024.4.11':
    df_original = load_data('kbars_MXF202412_2023-12-21-2024-04-11.pkl')
    product_name = '小台指期貨'

if choice == '英業達2020.1.2 至 2024.4.12':
    df_original = load_data('kbars_2356_2020-01-01-2024-04-12.pkl')
    product_name = '英業達2356'

if choice == '堤維西2020.1.2 至 2024.4.12':
    df_original = load_data('kbars_1522_2020-01-01-2024-04-12.pkl')
    product_name = '堤維西1522'


st.subheader("選擇資料時間區間")
if choice == '台積電: 2022.1.1 至 2024.4.9':
    date_range = st.date_input(
        '選擇日期區間 (2022-01-01 至 2024-04-09)',
        value=(datetime.date(2022, 1, 1), datetime.date(2024, 4, 9)),
        min_value=datetime.date(2022, 1, 1),
        max_value=datetime.date(2024, 4, 9)
    )

if choice == '大台指期貨2024.12到期: 2023.12 至 2024.4.11':
    date_range = st.date_input(
        '選擇日期區間 (2023-12-21 至 2024-04-11)',
        value=(datetime.date(2023, 12, 21), datetime.date(2024, 4, 11)),
        min_value=datetime.date(2023, 12, 21),
        max_value=datetime.date(2024, 4, 11)
    )

if choice == '小台指期貨2024.12到期: 2023.12 至 2024.4.11':
    date_range = st.date_input(
        '選擇日期區間 (2023-12-21 至 2024-04-11)',
        value=(datetime.date(2023, 12, 21), datetime.date(2024, 4, 11)),
        min_value=datetime.date(2023, 12, 21),
        max_value=datetime.date(2024, 4, 11)
    )

if choice == '英業達2020.1.2 至 2024.4.12':
    date_range = st.date_input(
        '選擇日期區間 (2020-01-02 至 2024-04-12)',
        value=(datetime.date(2020, 1, 2), datetime.date(2024, 4, 12)),
        min_value=datetime.date(2020, 1, 2),
        max_value=datetime.date(2024, 4, 12)
    )

if choice == '堤維西2020.1.2 至 2024.4.12':
    date_range = st.date_input(
        '選擇日期區間 (2020-01-02 至 2024-04-12)',
        value=(datetime.date(2020, 1, 2), datetime.date(2024, 4, 12)),
        min_value=datetime.date(2020, 1, 2),
        max_value=datetime.date(2024, 4, 12)
    )

if len(date_range) != 2:
    st.warning("請選擇完整的開始與結束日期。")
    st.stop()

start_date = datetime.datetime.combine(date_range[0], datetime.time.min)
end_date = datetime.datetime.combine(date_range[1], datetime.time.min)
df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)].copy()

if len(df) == 0:
    st.error("所選資料區間無資料。")
    st.stop()


#%%
####### (2) 轉化為字典 #######
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def To_Dictionary_1(df, product_name):
    KBar_dic = df.to_dict()

    KBar_open_list = list(KBar_dic['open'].values())
    KBar_dic['open'] = np.array(KBar_open_list, dtype=np.float64)

    KBar_dic['product'] = np.repeat(product_name, KBar_dic['open'].size)

    KBar_time_list = list(KBar_dic['time'].values())
    KBar_time_list = [i.to_pydatetime() for i in KBar_time_list]
    KBar_dic['time'] = np.array(KBar_time_list)

    KBar_low_list = list(KBar_dic['low'].values())
    KBar_dic['low'] = np.array(KBar_low_list, dtype=np.float64)

    KBar_high_list = list(KBar_dic['high'].values())
    KBar_dic['high'] = np.array(KBar_high_list, dtype=np.float64)

    KBar_close_list = list(KBar_dic['close'].values())
    KBar_dic['close'] = np.array(KBar_close_list, dtype=np.float64)

    KBar_volume_list = list(KBar_dic['volume'].values())
    KBar_dic['volume'] = np.array(KBar_volume_list)

    KBar_amount_list = list(KBar_dic['amount'].values())
    KBar_dic['amount'] = np.array(KBar_amount_list)

    return KBar_dic

KBar_dic = To_Dictionary_1(df, product_name)


#%%
####### (3) 改變 KBar 時間長度 #######
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def Change_Cycle(Date, cycle_duration, KBar_dic, product_name):
    KBar = indicator_forKBar_short.KBar(Date, cycle_duration)

    for i in range(KBar_dic['time'].size):
        time_ = KBar_dic['time'][i]
        open_price = KBar_dic['open'][i]
        close_price = KBar_dic['close'][i]
        low_price = KBar_dic['low'][i]
        high_price = KBar_dic['high'][i]
        qty = KBar_dic['volume'][i]
        KBar.AddPrice(time_, open_price, close_price, low_price, high_price, qty)

    KBar_dic_new = {}
    KBar_dic_new['time'] = KBar.TAKBar['time']
    KBar_dic_new['product'] = np.repeat(product_name, KBar_dic_new['time'].size)
    KBar_dic_new['open'] = KBar.TAKBar['open']
    KBar_dic_new['high'] = KBar.TAKBar['high']
    KBar_dic_new['low'] = KBar.TAKBar['low']
    KBar_dic_new['close'] = KBar.TAKBar['close']
    KBar_dic_new['volume'] = KBar.TAKBar['volume']

    return KBar_dic_new


Date = start_date.strftime("%Y-%m-%d")

st.subheader("設定技術指標視覺化圖形之相關參數:")
with st.expander("設定K棒相關參數:"):
    choices_unit = ['以分鐘為單位', '以日為單位', '以週為單位', '以月為單位']
    choice_unit = st.selectbox('選擇計算K棒時間長度之單位', choices_unit, index=1)

    if choice_unit == '以分鐘為單位':
        cycle_duration = float(st.number_input('輸入一根 K 棒的時間長度(單位:分鐘, 一日=1440分鐘)', value=1, key="KBar_duration_分"))
    if choice_unit == '以日為單位':
        cycle_duration = float(st.number_input('輸入一根 K 棒的時間長度(單位:日)', value=1, key="KBar_duration_日")) * 1440
    if choice_unit == '以週為單位':
        cycle_duration = float(st.number_input('輸入一根 K 棒的時間長度(單位:週)', value=1, key="KBar_duration_週")) * 7 * 1440
    if choice_unit == '以月為單位':
        cycle_duration = float(st.number_input('輸入一根 K 棒的時間長度(單位:月, 一月=30天)', value=1, key="KBar_duration_月")) * 30 * 1440

KBar_dic = Change_Cycle(Date, cycle_duration, KBar_dic, product_name)
KBar_df = pd.DataFrame(KBar_dic)

if len(KBar_df) == 0:
    st.error("變換 KBar 週期後無資料。")
    st.stop()


#%%
####### (4) 計算各種技術指標 #######
def find_last_nan_index(series):
    nan_indexes = series[::-1].index[series[::-1].apply(pd.isna)]
    if len(nan_indexes) > 0:
        return nan_indexes[0]
    return 0


@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def Calculate_MA(df, period=10):
    return df['close'].rolling(window=period).mean()


@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def Calculate_RSI(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def Calculate_Bollinger_Bands(df, period=20, num_std_dev=2):
    df = df.copy()
    df['SMA'] = df['close'].rolling(window=period).mean()
    df['Standard_Deviation'] = df['close'].rolling(window=period).std()
    df['Upper_Band'] = df['SMA'] + (df['Standard_Deviation'] * num_std_dev)
    df['Lower_Band'] = df['SMA'] - (df['Standard_Deviation'] * num_std_dev)
    return df


@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def Calculate_MACD(df, fast_period=12, slow_period=26, signal_period=9):
    df = df.copy()
    df['EMA_Fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_Slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    return df


with st.expander("設定長短移動平均線的 K棒 長度:"):
    LongMAPeriod = st.slider('設定計算長移動平均線(MA)的 K棒週期數目(整數, 例如 10)', 0, 100, 10, key='visualization_MA_long')
    ShortMAPeriod = st.slider('設定計算短移動平均線(MA)的 K棒週期數目(整數, 例如 2)', 0, 100, 2, key='visualization_MA_short')

KBar_df['MA_long'] = Calculate_MA(KBar_df, period=LongMAPeriod)
KBar_df['MA_short'] = Calculate_MA(KBar_df, period=ShortMAPeriod)
last_nan_index_MA = find_last_nan_index(KBar_df['MA_long'])

with st.expander("設定長短 RSI 的 K棒 長度:"):
    LongRSIPeriod = st.slider('設定計算長RSI的 K棒週期數目(整數, 例如 10)', 0, 1000, 10, key='visualization_RSI_long')
    ShortRSIPeriod = st.slider('設定計算短RSI的 K棒週期數目(整數, 例如 2)', 0, 1000, 2, key='visualization_RSI_short')

KBar_df['RSI_long'] = Calculate_RSI(KBar_df, LongRSIPeriod)
KBar_df['RSI_short'] = Calculate_RSI(KBar_df, ShortRSIPeriod)
KBar_df['RSI_Middle'] = np.array([50] * len(KBar_df['time']))
last_nan_index_RSI = find_last_nan_index(KBar_df['RSI_long'])

with st.expander("設定布林通道(Bollinger Band)相關參數:"):
    period = st.slider('設定計算布林通道(Bollinger Band)上中下三通道之K棒週期數目(整數, 例如 20)', 0, 100, 20, key='BB_period')
    num_std_dev = st.slider('設定計算布林通道(Bollinger Band)上中(或下中)通道之帶寬(例如 2 代表上中通道寬度為2倍的標準差)', 0, 100, 2, key='BB_heigh')

KBar_df = Calculate_Bollinger_Bands(KBar_df, period, num_std_dev)
last_nan_index_BB = find_last_nan_index(KBar_df['SMA'])

with st.expander("設定MACD三種週期的K棒長度:"):
    fast_period = st.slider('設定計算 MACD快速線的K棒週期數目(例如 12根日K)', 0, 100, 12, key='visualization_MACD_quick')
    slow_period = st.slider('設定計算 MACD慢速線的K棒週期數目(例如 26根日K)', 0, 100, 26, key='visualization_MACD_slow')
    signal_period = st.slider('設定計算 MACD訊號線的K棒週期數目(例如 9根日K)', 0, 100, 9, key='visualization_MACD_signal')

KBar_df = Calculate_MACD(KBar_df, fast_period, slow_period, signal_period)
last_nan_index_MACD = find_last_nan_index(KBar_df['MACD'])


#%%
####### (5) 畫圖 #######
st.subheader("技術指標視覺化圖形")

with st.expander("K線圖, 移動平均線"):
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Candlestick(
        x=KBar_df['time'],
        open=KBar_df['open'], high=KBar_df['high'],
        low=KBar_df['low'], close=KBar_df['close'], name='K線'
    ), secondary_y=True)

    fig1.add_trace(go.Bar(
        x=KBar_df['time'], y=KBar_df['volume'],
        name='成交量', marker=dict(color='black')
    ), secondary_y=False)

    fig1.add_trace(go.Scatter(
        x=KBar_df['time'][last_nan_index_MA + 1:],
        y=KBar_df['MA_long'][last_nan_index_MA + 1:],
        mode='lines', line=dict(color='orange', width=2),
        name=f'{LongMAPeriod}-根 K棒 移動平均線'
    ), secondary_y=True)

    fig1.add_trace(go.Scatter(
        x=KBar_df['time'][last_nan_index_MA + 1:],
        y=KBar_df['MA_short'][last_nan_index_MA + 1:],
        mode='lines', line=dict(color='pink', width=2),
        name=f'{ShortMAPeriod}-根 K棒 移動平均線'
    ), secondary_y=True)

    fig1.layout.yaxis2.showgrid = True
    st.plotly_chart(fig1, use_container_width=True)

with st.expander("長短 RSI"):
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(
        x=KBar_df['time'][last_nan_index_RSI + 1:],
        y=KBar_df['RSI_long'][last_nan_index_RSI + 1:],
        mode='lines', line=dict(color='red', width=2),
        name=f'{LongRSIPeriod}-根 K棒 移動 RSI'
    ), secondary_y=False)

    fig2.add_trace(go.Scatter(
        x=KBar_df['time'][last_nan_index_RSI + 1:],
        y=KBar_df['RSI_short'][last_nan_index_RSI + 1:],
        mode='lines', line=dict(color='blue', width=2),
        name=f'{ShortRSIPeriod}-根 K棒 移動 RSI'
    ), secondary_y=False)

    fig2.layout.yaxis2.showgrid = True
    st.plotly_chart(fig2, use_container_width=True)

with st.expander("K線圖,布林通道"):
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Candlestick(
        x=KBar_df['time'],
        open=KBar_df['open'], high=KBar_df['high'],
        low=KBar_df['low'], close=KBar_df['close'], name='K線'
    ), secondary_y=True)

    fig3.add_trace(go.Scatter(
        x=KBar_df['time'][last_nan_index_BB + 1:],
        y=KBar_df['SMA'][last_nan_index_BB + 1:],
        mode='lines', line=dict(color='black', width=2),
        name='布林通道中軌道'
    ), secondary_y=False)

    fig3.add_trace(go.Scatter(
        x=KBar_df['time'][last_nan_index_BB + 1:],
        y=KBar_df['Upper_Band'][last_nan_index_BB + 1:],
        mode='lines', line=dict(color='red', width=2),
        name='布林通道上軌道'
    ), secondary_y=False)

    fig3.add_trace(go.Scatter(
        x=KBar_df['time'][last_nan_index_BB + 1:],
        y=KBar_df['Lower_Band'][last_nan_index_BB + 1:],
        mode='lines', line=dict(color='blue', width=2),
        name='布林通道下軌道'
    ), secondary_y=False)

    fig3.layout.yaxis2.showgrid = True
    st.plotly_chart(fig3, use_container_width=True)

with st.expander("MACD(異同移動平均線)"):
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(go.Bar(
        x=KBar_df['time'], y=KBar_df['MACD_Histogram'],
        name='MACD Histogram', marker=dict(color='black')
    ), secondary_y=False)

    fig4.add_trace(go.Scatter(
        x=KBar_df['time'][last_nan_index_MACD + 1:],
        y=KBar_df['Signal_Line'][last_nan_index_MACD + 1:],
        mode='lines', line=dict(color='orange', width=2),
        name='訊號線(DEA)'
    ), secondary_y=True)

    fig4.add_trace(go.Scatter(
        x=KBar_df['time'][last_nan_index_MACD + 1:],
        y=KBar_df['MACD'][last_nan_index_MACD + 1:],
        mode='lines', line=dict(color='pink', width=2),
        name='DIF'
    ), secondary_y=True)

    fig4.layout.yaxis2.showgrid = True
    st.plotly_chart(fig4, use_container_width=True)


#%%
####### (6) 程式交易 #######
st.subheader("程式交易:")

def ChartOrder_MA(Kbar_df, TR, last_nan_index_MA_trading, LongMAPeriod_trading, ShortMAPeriod_trading):
    BTR = [i for i in TR if i[0] == 'Buy' or i[0] == 'B']
    BuyOrderPoint_date, BuyOrderPoint_price = [], []
    BuyCoverPoint_date, BuyCoverPoint_price = [], []

    for date, Low, High in zip(Kbar_df['time'], Kbar_df['low'], Kbar_df['high']):
        if date in [i[2] for i in BTR]:
            BuyOrderPoint_date.append(date)
            BuyOrderPoint_price.append(Low * 0.999)
        else:
            BuyOrderPoint_date.append(np.nan)
            BuyOrderPoint_price.append(np.nan)

        if date in [i[4] for i in BTR]:
            BuyCoverPoint_date.append(date)
            BuyCoverPoint_price.append(High * 1.001)
        else:
            BuyCoverPoint_date.append(np.nan)
            BuyCoverPoint_price.append(np.nan)

    STR = [i for i in TR if i[0] == 'Sell' or i[0] == 'S']
    SellOrderPoint_date, SellOrderPoint_price = [], []
    SellCoverPoint_date, SellCoverPoint_price = [], []

    for date, Low, High in zip(Kbar_df['time'], Kbar_df['low'], Kbar_df['high']):
        if date in [i[2] for i in STR]:
            SellOrderPoint_date.append(date)
            SellOrderPoint_price.append(High * 1.001)
        else:
            SellOrderPoint_date.append(np.nan)
            SellOrderPoint_price.append(np.nan)

        if date in [i[4] for i in STR]:
            SellCoverPoint_date.append(date)
            SellCoverPoint_price.append(Low * 0.999)
        else:
            SellCoverPoint_date.append(np.nan)
            SellCoverPoint_price.append(np.nan)

    fig5 = make_subplots(specs=[[{"secondary_y": True}]])
    fig5.add_trace(go.Scatter(
        x=Kbar_df['time'][last_nan_index_MA_trading + 1:],
        y=Kbar_df['MA_long'][last_nan_index_MA_trading + 1:],
        mode='lines', line=dict(color='orange', width=2),
        name=f'{LongMAPeriod_trading}-根 K棒 移動平均線'
    ), secondary_y=False)

    fig5.add_trace(go.Scatter(
        x=Kbar_df['time'][last_nan_index_MA_trading + 1:],
        y=Kbar_df['MA_short'][last_nan_index_MA_trading + 1:],
        mode='lines', line=dict(color='pink', width=2),
        name=f'{ShortMAPeriod_trading}-根 K棒 移動平均線'
    ), secondary_y=False)

    fig5.add_trace(go.Scatter(
        x=BuyOrderPoint_date, y=BuyOrderPoint_price,
        mode='markers', marker=dict(color='red', symbol='triangle-up', size=10),
        name='作多進場點'
    ), secondary_y=False)

    fig5.add_trace(go.Scatter(
        x=BuyCoverPoint_date, y=BuyCoverPoint_price,
        mode='markers', marker=dict(color='blue', symbol='triangle-down', size=10),
        name='作多出場點'
    ), secondary_y=False)

    fig5.add_trace(go.Scatter(
        x=SellOrderPoint_date, y=SellOrderPoint_price,
        mode='markers', marker=dict(color='green', symbol='triangle-down', size=10),
        name='作空進場點'
    ), secondary_y=False)

    fig5.add_trace(go.Scatter(
        x=SellCoverPoint_date, y=SellCoverPoint_price,
        mode='markers', marker=dict(color='black', symbol='triangle-up', size=10),
        name='作空出場點'
    ), secondary_y=False)

    fig5.layout.yaxis2.showgrid = True
    st.plotly_chart(fig5, use_container_width=True)


choices_strategy = ['<進場>: 移動平均線黃金交叉作多,死亡交叉作空. <出場>: 結算平倉(期貨), 移動停損.']
choice_strategy = st.selectbox('選擇交易策略', choices_strategy, index=0)

OrderRecord = Record()

if choice_strategy == '<進場>: 移動平均線黃金交叉作多,死亡交叉作空. <出場>: 結算平倉(期貨), 移動停損.':
    with st.expander("<策略參數設定>: 交易停損量、長移動平均線(MA)的K棒週期數目、短移動平均線(MA)的K棒週期數目、購買數量"):
        MoveStopLoss = st.slider(
            '選擇程式交易停損量(股票:每股價格; 期貨(大小台指):台股指數點數. 例如: 股票進場做多時, 取30代表停損價格為目前每股價格減30元; 大小台指進場做多時, 取30代表停損指數為目前台股指數減30點)',
            0, 100, 30, key='MoveStopLoss'
        )
        LongMAPeriod_trading = st.slider('設定計算長移動平均線(MA)的 K棒週期數目(整數, 例如 10)', 0, 100, 10, key='trading_MA_long')
        ShortMAPeriod_trading = st.slider('設定計算短移動平均線(MA)的 K棒週期數目(整數, 例如 2)', 0, 100, 2, key='trading_MA_short')
        Order_Quantity = st.slider('選擇購買數量(股票單位為張數(一張為1000股); 期貨單位為口數)', 1, 100, 1, key='Order_Quantity')

        KBar_df['MA_long'] = Calculate_MA(KBar_df, period=LongMAPeriod_trading)
        KBar_df['MA_short'] = Calculate_MA(KBar_df, period=ShortMAPeriod_trading)
        last_nan_index_MA_trading = find_last_nan_index(KBar_df['MA_long'])

    for n in range(1, len(KBar_df['time']) - 1):
        if not np.isnan(KBar_df['MA_long'][n - 1]):
            if OrderRecord.GetOpenInterest() == 0:
                if KBar_df['MA_short'][n - 1] <= KBar_df['MA_long'][n - 1] and KBar_df['MA_short'][n] > KBar_df['MA_long'][n]:
                    OrderRecord.Order('Buy', KBar_df['product'][n + 1], KBar_df['time'][n + 1], KBar_df['open'][n + 1], Order_Quantity)
                    OrderPrice = KBar_df['open'][n + 1]
                    StopLossPoint = OrderPrice - MoveStopLoss
                    continue

                if KBar_df['MA_short'][n - 1] >= KBar_df['MA_long'][n - 1] and KBar_df['MA_short'][n] < KBar_df['MA_long'][n]:
                    OrderRecord.Order('Sell', KBar_df['product'][n + 1], KBar_df['time'][n + 1], KBar_df['open'][n + 1], Order_Quantity)
                    OrderPrice = KBar_df['open'][n + 1]
                    StopLossPoint = OrderPrice + MoveStopLoss
                    continue

            elif OrderRecord.GetOpenInterest() > 0:
                if KBar_df['product'][n + 1] != KBar_df['product'][n]:
                    OrderRecord.Cover('Sell', KBar_df['product'][n], KBar_df['time'][n], KBar_df['close'][n], OrderRecord.GetOpenInterest())
                    continue

                if KBar_df['close'][n] - MoveStopLoss > StopLossPoint:
                    StopLossPoint = KBar_df['close'][n] - MoveStopLoss
                elif KBar_df['close'][n] < StopLossPoint:
                    OrderRecord.Cover('Sell', KBar_df['product'][n + 1], KBar_df['time'][n + 1], KBar_df['open'][n + 1], OrderRecord.GetOpenInterest())
                    continue

            elif OrderRecord.GetOpenInterest() < 0:
                if KBar_df['product'][n + 1] != KBar_df['product'][n]:
                    OrderRecord.Cover('Buy', KBar_df['product'][n], KBar_df['time'][n], KBar_df['close'][n], -OrderRecord.GetOpenInterest())
                    continue

                if KBar_df['close'][n] + MoveStopLoss < StopLossPoint:
                    StopLossPoint = KBar_df['close'][n] + MoveStopLoss
                elif KBar_df['close'][n] > StopLossPoint:
                    OrderRecord.Cover('Buy', KBar_df['product'][n + 1], KBar_df['time'][n + 1], KBar_df['open'][n + 1], -OrderRecord.GetOpenInterest())
                    continue

    ChartOrder_MA(KBar_df, OrderRecord.GetTradeRecord(), last_nan_index_MA_trading, LongMAPeriod_trading, ShortMAPeriod_trading)


###### 計算績效
交易總盈虧 = OrderRecord.GetTotalProfit()
平均每次盈虧 = OrderRecord.GetAverageProfit()
平均投資報酬率 = OrderRecord.GetAverageProfitRate()
平均獲利_只看獲利的 = OrderRecord.GetAverEarn()
平均虧損_只看虧損的 = OrderRecord.GetAverLoss()
勝率 = OrderRecord.GetWinRate()
最大連續虧損 = OrderRecord.GetAccLoss()
最大盈虧回落_MDD = OrderRecord.GetMDD()

if 最大盈虧回落_MDD > 0:
    報酬風險比 = 交易總盈虧 / 最大盈虧回落_MDD
else:
    報酬風險比 = '資料不足無法計算'

if len(OrderRecord.Profit) > 0:
    performance_data = {
        "項目": [
            "交易總盈虧(元)",
            "平均每次盈虧(元)",
            "平均投資報酬率",
            "平均獲利(只看獲利的)(元)",
            "平均虧損(只看虧損的)(元)",
            "勝率",
            "最大連續虧損(元)",
            "最大盈虧回落(MDD)(元)",
            "報酬風險比(交易總盈虧/最大盈虧回落(MDD))"
        ],
        "數值": [
            交易總盈虧,
            平均每次盈虧,
            平均投資報酬率,
            平均獲利_只看獲利的,
            平均虧損_只看虧損的,
            勝率,
            最大連續虧損,
            最大盈虧回落_MDD,
            報酬風險比
        ]
    }
    perf_df = pd.DataFrame(performance_data)
    st.write(perf_df)
else:
    st.write('沒有交易記錄(已經了結之交易) !')


##### 畫累計盈虧圖
OrderRecord.GeneratorProfitChart(StrategyName='MA')

##### 畫累計投資報酬率圖
OrderRecord.GeneratorProfit_rateChart(StrategyName='MA')


#%%
####### (五) MACD 策略 #######
st.subheader("(五) MACD 策略")
st.markdown("""
**進場規則：**
- 多方進場：DIF（快線）由下往上穿越 DEA（慢線），即柱狀體由負轉正（黃金交叉）
- 空方進場：DIF 由上往下穿越 DEA，即柱狀體由正轉負（死亡交叉）

**出場規則：** 移動停損
""")

OrderRecord_MACD = Record()

with st.expander("<MACD策略參數設定>"):
    MACD_fast = st.slider('MACD 快線週期（fast period）', 2, 50, 12, key='macd_fast')
    MACD_slow = st.slider('MACD 慢線週期（slow period）', 3, 100, 26, key='macd_slow')
    MACD_signal = st.slider('MACD 訊號線週期（signal period）', 2, 30, 9, key='macd_signal')
    MACD_StopLoss = st.slider('移動停損量（股票:每股元；期貨:點數）', 0, 200, 30, key='macd_stoploss')
    MACD_Qty = st.slider('購買數量（股票:張；期貨:口）', 1, 100, 1, key='macd_qty')

KBar_df_macd = KBar_df.copy()
macd_result = MACD(KBar_df_macd.to_dict('series'), fastperiod=MACD_fast, slowperiod=MACD_slow, signalperiod=MACD_signal)
if isinstance(macd_result, (list, tuple)):
    KBar_df_macd['macd'] = np.asarray(macd_result[0], dtype=np.float64)
    KBar_df_macd['macdsignal'] = np.asarray(macd_result[1], dtype=np.float64)
    KBar_df_macd['macdhist'] = np.asarray(macd_result[2], dtype=np.float64)
else:
    KBar_df_macd['macd'] = macd_result['macd'].values
    KBar_df_macd['macdsignal'] = macd_result['macdsignal'].values
    KBar_df_macd['macdhist'] = macd_result['macdhist'].values

KBar_df_macd = KBar_df_macd.reset_index(drop=True)

for n in range(1, len(KBar_df_macd) - 1):
    if np.isnan(KBar_df_macd['macdhist'][n - 1]) or np.isnan(KBar_df_macd['macdhist'][n]):
        continue
    if OrderRecord_MACD.GetOpenInterest() == 0:
        if KBar_df_macd['macdhist'][n - 1] < 0 and KBar_df_macd['macdhist'][n] >= 0:
            OrderRecord_MACD.Order('Buy', KBar_df_macd['product'][n+1], KBar_df_macd['time'][n+1], KBar_df_macd['open'][n+1], MACD_Qty)
            StopLossPoint_MACD = KBar_df_macd['open'][n+1] - MACD_StopLoss
            continue
        if KBar_df_macd['macdhist'][n - 1] > 0 and KBar_df_macd['macdhist'][n] <= 0:
            OrderRecord_MACD.Order('Sell', KBar_df_macd['product'][n+1], KBar_df_macd['time'][n+1], KBar_df_macd['open'][n+1], MACD_Qty)
            StopLossPoint_MACD = KBar_df_macd['open'][n+1] + MACD_StopLoss
            continue
    elif OrderRecord_MACD.GetOpenInterest() > 0:
        if KBar_df_macd['product'][n+1] != KBar_df_macd['product'][n]:
            OrderRecord_MACD.Cover('Sell', KBar_df_macd['product'][n], KBar_df_macd['time'][n], KBar_df_macd['close'][n], OrderRecord_MACD.GetOpenInterest())
            continue
        if KBar_df_macd['close'][n] - MACD_StopLoss > StopLossPoint_MACD:
            StopLossPoint_MACD = KBar_df_macd['close'][n] - MACD_StopLoss
        elif KBar_df_macd['close'][n] < StopLossPoint_MACD:
            OrderRecord_MACD.Cover('Sell', KBar_df_macd['product'][n+1], KBar_df_macd['time'][n+1], KBar_df_macd['open'][n+1], OrderRecord_MACD.GetOpenInterest())
            continue
    elif OrderRecord_MACD.GetOpenInterest() < 0:
        if KBar_df_macd['product'][n+1] != KBar_df_macd['product'][n]:
            OrderRecord_MACD.Cover('Buy', KBar_df_macd['product'][n], KBar_df_macd['time'][n], KBar_df_macd['close'][n], -OrderRecord_MACD.GetOpenInterest())
            continue
        if KBar_df_macd['close'][n] + MACD_StopLoss < StopLossPoint_MACD:
            StopLossPoint_MACD = KBar_df_macd['close'][n] + MACD_StopLoss
        elif KBar_df_macd['close'][n] > StopLossPoint_MACD:
            OrderRecord_MACD.Cover('Buy', KBar_df_macd['product'][n+1], KBar_df_macd['time'][n+1], KBar_df_macd['open'][n+1], -OrderRecord_MACD.GetOpenInterest())
            continue

if len(OrderRecord_MACD.Profit) > 0:
    macd_total = OrderRecord_MACD.GetTotalProfit()
    macd_mdd = OrderRecord_MACD.GetMDD()
    perf_macd = pd.DataFrame({
        "項目": ["交易總盈虧(元)", "平均每次盈虧(元)", "平均投資報酬率", "勝率", "最大連續虧損(元)", "最大盈虧回落MDD(元)", "報酬風險比"],
        "數值": [
            macd_total,
            OrderRecord_MACD.GetAverageProfit(),
            OrderRecord_MACD.GetAverageProfitRate(),
            OrderRecord_MACD.GetWinRate(),
            OrderRecord_MACD.GetAccLoss(),
            macd_mdd,
            macd_total / macd_mdd if macd_mdd > 0 else 'N/A'
        ]
    })
    st.write(perf_macd)
    OrderRecord_MACD.GeneratorProfitChart(StrategyName='MACD')
    OrderRecord_MACD.GeneratorProfit_rateChart(StrategyName='MACD')
else:
    st.write('MACD 策略：沒有交易記錄！')


#%%
####### (六) KDJ 策略 #######
st.subheader("(六) KDJ 策略")
st.markdown("""
**進場規則：**
- 多方進場：K 由下往上穿越 D（黃金交叉），且 K < 超賣線
- 空方進場：K 由上往下穿越 D（死亡交叉），且 K > 超買線

**出場規則：** 移動停損
""")

OrderRecord_KDJ = Record()

with st.expander("<KDJ策略參數設定>"):
    KDJ_fastk = st.slider('RSV 計算週期（fastk_period）', 3, 30, 9, key='kdj_fastk')
    KDJ_slowk = st.slider('K 值平滑週期（slowk_period）', 1, 10, 3, key='kdj_slowk')
    KDJ_slowd = st.slider('D 值平滑週期（slowd_period）', 1, 10, 3, key='kdj_slowd')
    KDJ_overbought = st.slider('超買線（K > 此值時為超買）', 50, 95, 80, key='kdj_overbought')
    KDJ_oversold = st.slider('超賣線（K < 此值時為超賣）', 5, 50, 20, key='kdj_oversold')
    KDJ_StopLoss = st.slider('移動停損量（股票:每股元；期貨:點數）', 0, 200, 30, key='kdj_stoploss')
    KDJ_Qty = st.slider('購買數量（股票:張；期貨:口）', 1, 100, 1, key='kdj_qty')

KBar_df_kdj = KBar_df.copy().reset_index(drop=True)
stoch_result = STOCH(
    KBar_df_kdj.to_dict('series'),
    fastk_period=KDJ_fastk,
    slowk_period=KDJ_slowk,
    slowk_matype=0,
    slowd_period=KDJ_slowd,
    slowd_matype=0
)
if isinstance(stoch_result, (list, tuple)):
    KBar_df_kdj['slowk'] = np.asarray(stoch_result[0], dtype=np.float64)
    KBar_df_kdj['slowd'] = np.asarray(stoch_result[1], dtype=np.float64)
else:
    KBar_df_kdj['slowk'] = stoch_result['slowk'].values
    KBar_df_kdj['slowd'] = stoch_result['slowd'].values
KBar_df_kdj['J'] = 3 * KBar_df_kdj['slowk'] - 2 * KBar_df_kdj['slowd']

for n in range(1, len(KBar_df_kdj) - 1):
    if np.isnan(KBar_df_kdj['slowk'][n - 1]) or np.isnan(KBar_df_kdj['slowk'][n]):
        continue
    if OrderRecord_KDJ.GetOpenInterest() == 0:
        if (KBar_df_kdj['slowk'][n - 1] <= KBar_df_kdj['slowd'][n - 1] and
                KBar_df_kdj['slowk'][n] > KBar_df_kdj['slowd'][n] and
                KBar_df_kdj['slowk'][n] < KDJ_oversold):
            OrderRecord_KDJ.Order('Buy', KBar_df_kdj['product'][n+1], KBar_df_kdj['time'][n+1], KBar_df_kdj['open'][n+1], KDJ_Qty)
            StopLossPoint_KDJ = KBar_df_kdj['open'][n+1] - KDJ_StopLoss
            continue
        if (KBar_df_kdj['slowk'][n - 1] >= KBar_df_kdj['slowd'][n - 1] and
                KBar_df_kdj['slowk'][n] < KBar_df_kdj['slowd'][n] and
                KBar_df_kdj['slowk'][n] > KDJ_overbought):
            OrderRecord_KDJ.Order('Sell', KBar_df_kdj['product'][n+1], KBar_df_kdj['time'][n+1], KBar_df_kdj['open'][n+1], KDJ_Qty)
            StopLossPoint_KDJ = KBar_df_kdj['open'][n+1] + KDJ_StopLoss
            continue
    elif OrderRecord_KDJ.GetOpenInterest() > 0:
        if KBar_df_kdj['product'][n+1] != KBar_df_kdj['product'][n]:
            OrderRecord_KDJ.Cover('Sell', KBar_df_kdj['product'][n], KBar_df_kdj['time'][n], KBar_df_kdj['close'][n], OrderRecord_KDJ.GetOpenInterest())
            continue
        if KBar_df_kdj['close'][n] - KDJ_StopLoss > StopLossPoint_KDJ:
            StopLossPoint_KDJ = KBar_df_kdj['close'][n] - KDJ_StopLoss
        elif KBar_df_kdj['close'][n] < StopLossPoint_KDJ:
            OrderRecord_KDJ.Cover('Sell', KBar_df_kdj['product'][n+1], KBar_df_kdj['time'][n+1], KBar_df_kdj['open'][n+1], OrderRecord_KDJ.GetOpenInterest())
            continue
    elif OrderRecord_KDJ.GetOpenInterest() < 0:
        if KBar_df_kdj['product'][n+1] != KBar_df_kdj['product'][n]:
            OrderRecord_KDJ.Cover('Buy', KBar_df_kdj['product'][n], KBar_df_kdj['time'][n], KBar_df_kdj['close'][n], -OrderRecord_KDJ.GetOpenInterest())
            continue
        if KBar_df_kdj['close'][n] + KDJ_StopLoss < StopLossPoint_KDJ:
            StopLossPoint_KDJ = KBar_df_kdj['close'][n] + KDJ_StopLoss
        elif KBar_df_kdj['close'][n] > StopLossPoint_KDJ:
            OrderRecord_KDJ.Cover('Buy', KBar_df_kdj['product'][n+1], KBar_df_kdj['time'][n+1], KBar_df_kdj['open'][n+1], -OrderRecord_KDJ.GetOpenInterest())
            continue

if len(OrderRecord_KDJ.Profit) > 0:
    kdj_total = OrderRecord_KDJ.GetTotalProfit()
    kdj_mdd = OrderRecord_KDJ.GetMDD()
    perf_kdj = pd.DataFrame({
        "項目": ["交易總盈虧(元)", "平均每次盈虧(元)", "平均投資報酬率", "勝率", "最大連續虧損(元)", "最大盈虧回落MDD(元)", "報酬風險比"],
        "數值": [
            kdj_total,
            OrderRecord_KDJ.GetAverageProfit(),
            OrderRecord_KDJ.GetAverageProfitRate(),
            OrderRecord_KDJ.GetWinRate(),
            OrderRecord_KDJ.GetAccLoss(),
            kdj_mdd,
            kdj_total / kdj_mdd if kdj_mdd > 0 else 'N/A'
        ]
    })
    st.write(perf_kdj)
    OrderRecord_KDJ.GeneratorProfitChart(StrategyName='KDJ')
    OrderRecord_KDJ.GeneratorProfit_rateChart(StrategyName='KDJ')
else:
    st.write('KDJ 策略：沒有交易記錄！')


#%%
####### (七) 策略參數最佳化 #######
st.subheader("(七) 策略參數最佳化")
st.markdown("對所選策略進行 Grid Search，同時考慮**報酬（總盈虧）**與**風險（MDD）**，找出最佳參數組合。")

def run_ma_backtest(df, long_p, short_p, stop_loss, qty):
    rec = Record()
    df = df.copy().reset_index(drop=True)
    df['MA_long'] = df['close'].rolling(long_p).mean()
    df['MA_short'] = df['close'].rolling(short_p).mean()
    for n in range(1, len(df) - 1):
        if pd.isna(df['MA_long'][n-1]):
            continue
        if rec.GetOpenInterest() == 0:
            if df['MA_short'][n-1] <= df['MA_long'][n-1] and df['MA_short'][n] > df['MA_long'][n]:
                rec.Order('Buy', df['product'][n+1], df['time'][n+1], df['open'][n+1], qty)
                sl = df['open'][n+1] - stop_loss
                continue
            if df['MA_short'][n-1] >= df['MA_long'][n-1] and df['MA_short'][n] < df['MA_long'][n]:
                rec.Order('Sell', df['product'][n+1], df['time'][n+1], df['open'][n+1], qty)
                sl = df['open'][n+1] + stop_loss
                continue
        elif rec.GetOpenInterest() > 0:
            if df['product'][n+1] != df['product'][n]:
                rec.Cover('Sell', df['product'][n], df['time'][n], df['close'][n], rec.GetOpenInterest())
                continue
            if df['close'][n] - stop_loss > sl:
                sl = df['close'][n] - stop_loss
            elif df['close'][n] < sl:
                rec.Cover('Sell', df['product'][n+1], df['time'][n+1], df['open'][n+1], rec.GetOpenInterest())
                continue
        elif rec.GetOpenInterest() < 0:
            if df['product'][n+1] != df['product'][n]:
                rec.Cover('Buy', df['product'][n], df['time'][n], df['close'][n], -rec.GetOpenInterest())
                continue
            if df['close'][n] + stop_loss < sl:
                sl = df['close'][n] + stop_loss
            elif df['close'][n] > sl:
                rec.Cover('Buy', df['product'][n+1], df['time'][n+1], df['open'][n+1], -rec.GetOpenInterest())
                continue
    return rec

def run_macd_backtest(df, fast_p, slow_p, signal_p, stop_loss, qty):
    rec = Record()
    df = df.copy().reset_index(drop=True)
    res = MACD(df.to_dict('series'), fastperiod=fast_p, slowperiod=slow_p, signalperiod=signal_p)
    hist = np.asarray(res[0] if isinstance(res, (list, tuple)) else res['macdhist'], dtype=np.float64)
    for n in range(1, len(df) - 1):
        if np.isnan(hist[n-1]) or np.isnan(hist[n]):
            continue
        if rec.GetOpenInterest() == 0:
            if hist[n-1] < 0 and hist[n] >= 0:
                rec.Order('Buy', df['product'][n+1], df['time'][n+1], df['open'][n+1], qty)
                sl = df['open'][n+1] - stop_loss
                continue
            if hist[n-1] > 0 and hist[n] <= 0:
                rec.Order('Sell', df['product'][n+1], df['time'][n+1], df['open'][n+1], qty)
                sl = df['open'][n+1] + stop_loss
                continue
        elif rec.GetOpenInterest() > 0:
            if df['product'][n+1] != df['product'][n]:
                rec.Cover('Sell', df['product'][n], df['time'][n], df['close'][n], rec.GetOpenInterest())
                continue
            if df['close'][n] - stop_loss > sl:
                sl = df['close'][n] - stop_loss
            elif df['close'][n] < sl:
                rec.Cover('Sell', df['product'][n+1], df['time'][n+1], df['open'][n+1], rec.GetOpenInterest())
                continue
        elif rec.GetOpenInterest() < 0:
            if df['product'][n+1] != df['product'][n]:
                rec.Cover('Buy', df['product'][n], df['time'][n], df['close'][n], -rec.GetOpenInterest())
                continue
            if df['close'][n] + stop_loss < sl:
                sl = df['close'][n] + stop_loss
            elif df['close'][n] > sl:
                rec.Cover('Buy', df['product'][n+1], df['time'][n+1], df['open'][n+1], -rec.GetOpenInterest())
                continue
    return rec

opt_strategy = st.selectbox('選擇要最佳化的策略', ['MA 移動平均線', 'MACD'], key='opt_strategy')

if opt_strategy == 'MA 移動平均線':
    with st.expander("設定 MA 最佳化參數範圍"):
        ma_long_range = st.slider('長均線範圍', 5, 60, (5, 30), key='opt_ma_long')
        ma_short_range = st.slider('短均線範圍', 1, 20, (1, 10), key='opt_ma_short')
        ma_sl_range = st.slider('停損範圍', 5, 100, (10, 50), key='opt_ma_sl')
        ma_step = st.number_input('步長（每隔幾格搜尋一次）', min_value=1, max_value=10, value=5, key='opt_ma_step')

    if st.button('執行 MA 最佳化', key='run_ma_opt'):
        results = []
        long_vals = range(ma_long_range[0], ma_long_range[1]+1, int(ma_step))
        short_vals = range(ma_short_range[0], ma_short_range[1]+1, int(ma_step))
        sl_vals = range(ma_sl_range[0], ma_sl_range[1]+1, int(ma_step))
        total_combos = sum(1 for lp in long_vals for sp in short_vals for sl in sl_vals if lp > sp)
        prog = st.progress(0, text='最佳化進行中...')
        done = 0
        for lp, sp, sl in itertools.product(long_vals, short_vals, sl_vals):
            if lp <= sp:
                continue
            rec = run_ma_backtest(KBar_df, lp, sp, sl, 1)
            tp = rec.GetTotalProfit()
            mdd = rec.GetMDD()
            wr = rec.GetWinRate()
            rr = tp / mdd if mdd > 0 else 0
            results.append({'長均線': lp, '短均線': sp, '停損': sl, '總盈虧': round(tp, 0), '勝率': round(wr, 3), 'MDD': round(mdd, 0), '報酬風險比': round(rr, 3)})
            done += 1
            prog.progress(min(done / max(total_combos, 1), 1.0), text=f'最佳化進行中... {done}/{total_combos}')
        if results:
            opt_df = pd.DataFrame(results).sort_values('報酬風險比', ascending=False).reset_index(drop=True)
            st.write("**最佳化結果（依報酬風險比排序）：**")
            st.dataframe(opt_df.head(20))
        else:
            st.write('無有效參數組合（請確認長均線 > 短均線）')

elif opt_strategy == 'MACD':
    with st.expander("設定 MACD 最佳化參數範圍"):
        macd_fast_range = st.slider('快線範圍', 2, 30, (5, 15), key='opt_macd_fast')
        macd_slow_range = st.slider('慢線範圍', 10, 60, (20, 35), key='opt_macd_slow')
        macd_sig_range = st.slider('訊號線範圍', 2, 20, (5, 12), key='opt_macd_sig')
        macd_sl_range = st.slider('停損範圍', 5, 100, (10, 50), key='opt_macd_sl')
        macd_step = st.number_input('步長', min_value=1, max_value=10, value=5, key='opt_macd_step')

    if st.button('執行 MACD 最佳化', key='run_macd_opt'):
        results = []
        fast_vals = range(macd_fast_range[0], macd_fast_range[1]+1, int(macd_step))
        slow_vals = range(macd_slow_range[0], macd_slow_range[1]+1, int(macd_step))
        sig_vals = range(macd_sig_range[0], macd_sig_range[1]+1, int(macd_step))
        sl_vals = range(macd_sl_range[0], macd_sl_range[1]+1, int(macd_step))
        prog = st.progress(0, text='MACD 最佳化進行中...')
        done = 0
        combos = [(f, s, sg, sl) for f in fast_vals for s in slow_vals for sg in sig_vals for sl in sl_vals if f < s]
        for f, s, sg, sl in combos:
            rec = run_macd_backtest(KBar_df, f, s, sg, sl, 1)
            tp = rec.GetTotalProfit()
            mdd = rec.GetMDD()
            wr = rec.GetWinRate()
            rr = tp / mdd if mdd > 0 else 0
            results.append({'快線': f, '慢線': s, '訊號線': sg, '停損': sl, '總盈虧': round(tp, 0), '勝率': round(wr, 3), 'MDD': round(mdd, 0), '報酬風險比': round(rr, 3)})
            done += 1
            prog.progress(min(done / max(len(combos), 1), 1.0), text=f'MACD 最佳化進行中... {done}/{len(combos)}')
        if results:
            opt_df = pd.DataFrame(results).sort_values('報酬風險比', ascending=False).reset_index(drop=True)
            st.write("**最佳化結果（依報酬風險比排序）：**")
            st.dataframe(opt_df.head(20))
        else:
            st.write('無有效參數組合')


#%%
####### (八) 生成式 AI 自動評估策略績效 #######
st.subheader("(八) 生成式 AI 自動評估策略績效")
st.markdown("輸入 Anthropic API Key，讓 AI 自動分析並比較各策略的績效與風險。")

api_key_input = st.text_input('輸入 Anthropic API Key', type='password', key='ai_api_key')

def build_perf_summary(name, rec):
    if len(rec.Profit) == 0:
        return f"{name}: 無交易記錄"
    tp = rec.GetTotalProfit()
    mdd = rec.GetMDD()
    return (
        f"策略名稱: {name}\n"
        f"  交易次數: {rec.GetTotalNumber()}\n"
        f"  總盈虧: {tp:.0f} 元\n"
        f"  平均每次盈虧: {rec.GetAverageProfit():.0f} 元\n"
        f"  勝率: {rec.GetWinRate():.2%}\n"
        f"  最大盈虧回落(MDD): {mdd:.0f} 元\n"
        f"  報酬風險比(總盈虧/MDD): {tp/mdd:.2f}" if mdd > 0 else f"  報酬風險比: N/A"
    )

if st.button('讓 AI 評估策略', key='ai_eval'):
    if not api_key_input.strip():
        st.warning('請先輸入 Anthropic API Key')
    else:
        summary_ma = build_perf_summary('MA 移動平均線策略', OrderRecord)
        summary_macd = build_perf_summary('MACD 策略', OrderRecord_MACD)
        summary_kdj = build_perf_summary('KDJ 策略', OrderRecord_KDJ)

        prompt_text = f"""
以下是三種程式交易策略在同一金融商品（{product_name}）、同一時間區間的回測績效數據：

{summary_ma}

{summary_macd}

{summary_kdj}

請以專業的量化交易分析師角色，完成以下任務：
1. 逐一評估每個策略的優缺點（重點：報酬、風險、勝率、穩定性）
2. 比較三個策略，指出哪個策略最適合此商品與時間區間，並說明原因
3. 針對表現較差的策略，給出改進建議（例如：參數調整方向、加入過濾條件等）
4. 最後給出一個綜合建議（中文回答）
"""
        with st.spinner('AI 正在分析中...'):
            try:
                client = anthropic.Anthropic(api_key=api_key_input.strip())
                message = client.messages.create(
                    model='claude-opus-4-5',
                    max_tokens=1024,
                    messages=[{'role': 'user', 'content': prompt_text}]
                )
                ai_response = message.content[0].text
                st.markdown("### AI 評估結果")
                st.markdown(ai_response)
            except Exception as e:
                st.error(f'API 呼叫失敗：{e}')


#%%
####### (9) 呈現即時資料 #######