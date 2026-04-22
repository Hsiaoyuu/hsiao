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
from talib.abstract import SMA, EMA, WMA, RSI, BBANDS, MACD
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
    start_date = st.text_input('輸入開始日期(日期格式: 2022-01-01), 區間:2022-01-01 至 2024-04-09', '2022-01-01')
    end_date = st.text_input('輸入結束日期 (日期格式: 2024-04-09), 區間:2022-01-01 至 2024-04-09', '2024-04-09')

if choice == '大台指期貨2024.12到期: 2023.12 至 2024.4.11':
    start_date = st.text_input('輸入開始日期(日期格式: 2023-12-21), 區間:2023-12-21 至 2024-04-11', '2023-12-21')
    end_date = st.text_input('輸入結束日期 (日期格式: 2024-04-11), 區間:2023-12-21 至 2024-04-11', '2024-04-11')

if choice == '小台指期貨2024.12到期: 2023.12 至 2024.4.11':
    start_date = st.text_input('輸入開始日期(日期格式: 2023-12-21), 區間:2023-12-21 至 2024-04-11', '2023-12-21')
    end_date = st.text_input('輸入結束日期 (日期格式: 2024-04-11), 區間:2023-12-21 至 2024-04-11', '2024-04-11')

if choice == '英業達2020.1.2 至 2024.4.12':
    start_date = st.text_input('輸入開始日期(日期格式: 2020-01-02), 區間:2020-01-02 至 2024-04-12', '2020-01-02')
    end_date = st.text_input('輸入結束日期 (日期格式: 2024-04-12), 區間:2020-01-02 至 2024-04-12', '2024-04-12')

if choice == '堤維西2020.1.2 至 2024.4.12':
    start_date = st.text_input('輸入開始日期(日期格式: 2020-01-02), 區間:2020-01-02 至 2024-04-12', '2020-01-02')
    end_date = st.text_input('輸入結束日期 (日期格式: 2024-04-12), 區間:2020-01-02 至 2024-04-12', '2024-04-12')

start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
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
####### (7) 呈現即時資料 #######