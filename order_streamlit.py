# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:20:15 2026

@author: user
"""

# 載入必要套件
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import streamlit as st


class Record():
    """
    最終版設計原則
    ----------------
    1. 股票數量單位固定為「張」，1張 = 1000股
    2. 期貨數量單位固定為「口」
    3. Profit 一律直接記錄為 TWD
    4. Profit_rate 一律記錄為單筆交易報酬率(無單位)
    5. 同一次多數量交易只記成一筆交易，不再拆成多筆
    """

    STOCK_LOT_SIZE = 1000

    FUTURE_MULTIPLIER_MAP = {
        '大台指期貨': 200,
        '小台指期貨': 50,
        '微台指期貨': 10,
        'TXF': 200,
        'MXF': 50,
        'TMF': 10,
    }

    def __init__(self):
        # 儲存績效
        self.Profit = []         # 每筆完整交易總損益(TWD)
        self.Profit_rate = []    # 每筆完整交易報酬率

        # 未平倉
        self.OpenInterestQty = 0
        self.OpenInterest = []   # [side, product, order_time, order_price, qty]

        # 交易紀錄總計
        self.TradeRecord = []    # [B/S, product, entry_time, entry_price, exit_time, exit_price, qty]

    # ----------------------------
    # 內部工具函式
    # ----------------------------
    def _get_contract_multiplier(self, product):
        return self.FUTURE_MULTIPLIER_MAP.get(product, 1)

    def _is_future_product(self, product):
        return product in self.FUTURE_MULTIPLIER_MAP

    def _calc_trade_profit_twd_and_rate(self, side, product, entry_price, exit_price, qty):
        """
        side:
            'B' => 多單交易
            'S' => 空單交易

        股票:
            qty 單位 = 張
            TWD損益 = 每股價差 * 1000 * 張數

        期貨:
            qty 單位 = 口
            TWD損益 = 點差 * 每點價值 * 口數
        """
        if self._is_future_product(product):
            multiplier = self._get_contract_multiplier(product)
            unit_notional = entry_price * multiplier
            if side == 'B':
                profit_twd = (exit_price - entry_price) * multiplier * qty
            else:
                profit_twd = (entry_price - exit_price) * multiplier * qty

            denominator = unit_notional * qty if abs(unit_notional * qty) > 1e-12 else 1e-12
            profit_rate = profit_twd / denominator
            return profit_twd, profit_rate

        # 股票: qty = 張
        shares = qty * self.STOCK_LOT_SIZE
        unit_notional = entry_price * shares

        if side == 'B':
            profit_twd = (exit_price - entry_price) * shares
        else:
            profit_twd = (entry_price - exit_price) * shares

        denominator = unit_notional if abs(unit_notional) > 1e-12 else 1e-12
        profit_rate = profit_twd / denominator
        return profit_twd, profit_rate

    # ----------------------------
    # 進場
    # ----------------------------
    def Order(self, BS, Product, OrderTime, OrderPrice, OrderQty):
        if OrderQty <= 0:
            return

        qty = int(OrderQty)

        if BS == 'B' or BS == 'Buy':
            self.OpenInterest.append([1, Product, OrderTime, OrderPrice, qty])
            self.OpenInterestQty += qty

        elif BS == 'S' or BS == 'Sell':
            self.OpenInterest.append([-1, Product, OrderTime, OrderPrice, qty])
            self.OpenInterestQty -= qty

    # ----------------------------
    # 出場
    # ----------------------------
    def Cover(self, BS, Product, CoverTime, CoverPrice, CoverQty):
        if CoverQty <= 0:
            return

        remain_qty = int(CoverQty)

        # 平多單
        if BS == 'S' or BS == 'Sell':
            while remain_qty > 0:
                long_positions = [x for x in self.OpenInterest if x[0] == 1]
                if len(long_positions) == 0:
                    print('尚無進場')
                    return

                pos = long_positions[0]
                pos_qty = pos[4]
                close_qty = min(remain_qty, pos_qty)

                entry_price = pos[3]
                trade_product = pos[1]

                profit_twd, profit_rate = self._calc_trade_profit_twd_and_rate(
                    side='B',
                    product=trade_product,
                    entry_price=entry_price,
                    exit_price=CoverPrice,
                    qty=close_qty
                )

                self.TradeRecord.append([
                    'B',
                    pos[1],
                    pos[2],
                    pos[3],
                    CoverTime,
                    CoverPrice,
                    close_qty
                ])

                self.Profit.append(profit_twd)
                self.Profit_rate.append(profit_rate)

                if close_qty == pos_qty:
                    self.OpenInterest.remove(pos)
                else:
                    pos[4] -= close_qty

                self.OpenInterestQty -= close_qty
                remain_qty -= close_qty

        # 平空單
        elif BS == 'B' or BS == 'Buy':
            while remain_qty > 0:
                short_positions = [x for x in self.OpenInterest if x[0] == -1]
                if len(short_positions) == 0:
                    print('尚無進場')
                    return

                pos = short_positions[0]
                pos_qty = pos[4]
                close_qty = min(remain_qty, pos_qty)

                entry_price = pos[3]
                trade_product = pos[1]

                profit_twd, profit_rate = self._calc_trade_profit_twd_and_rate(
                    side='S',
                    product=trade_product,
                    entry_price=entry_price,
                    exit_price=CoverPrice,
                    qty=close_qty
                )

                self.TradeRecord.append([
                    'S',
                    pos[1],
                    pos[2],
                    pos[3],
                    CoverTime,
                    CoverPrice,
                    close_qty
                ])

                self.Profit.append(profit_twd)
                self.Profit_rate.append(profit_rate)

                if close_qty == pos_qty:
                    self.OpenInterest.remove(pos)
                else:
                    pos[4] -= close_qty

                self.OpenInterestQty += close_qty
                remain_qty -= close_qty

    # ----------------------------
    # 基本查詢
    # ----------------------------
    def GetOpenInterest(self):
        return self.OpenInterestQty

    def GetTradeRecord(self):
        return self.TradeRecord

    def GetProfit(self):
        return self.Profit

    def GetProfitRate(self):
        return self.Profit_rate

    # ----------------------------
    # 績效統計
    # ----------------------------
    def GetTotalProfit(self):
        return sum(self.Profit) if len(self.Profit) > 0 else 0

    def GetTotalNumber(self):
        return len(self.Profit)

    def GetAverageProfit(self):
        return sum(self.Profit) / len(self.Profit) if len(self.Profit) > 0 else 0

    def GetAverageProfitRate(self):
        return sum(self.Profit_rate) / len(self.Profit_rate) if len(self.Profit_rate) > 0 else 0

    def GetWinRate(self):
        if len(self.Profit) == 0:
            return 0
        WinProfit = [i for i in self.Profit if i > 0]
        return len(WinProfit) / len(self.Profit)

    def GetAccLoss(self):
        if len(self.Profit) == 0:
            return 0

        AccLoss = 0
        MaxAccLoss = 0
        for p in self.Profit:
            if p <= 0:
                AccLoss += p
                if AccLoss < MaxAccLoss:
                    MaxAccLoss = AccLoss
            else:
                AccLoss = 0
        return MaxAccLoss

    def GetMDD(self):
        if len(self.Profit) == 0:
            return 0

        MDD, Capital, MaxCapital = 0, 0, 0
        for p in self.Profit:
            Capital += p
            MaxCapital = max(MaxCapital, Capital)
            DD = MaxCapital - Capital
            MDD = max(MDD, DD)
        return MDD

    def GetMDD_rate(self):
        if len(self.Profit_rate) == 0:
            return 0

        MDD_rate, Capital_rate, MaxCapital_rate = 0, 0, 0
        for p in self.Profit_rate:
            Capital_rate += p
            MaxCapital_rate = max(MaxCapital_rate, Capital_rate)
            DD_rate = MaxCapital_rate - Capital_rate
            MDD_rate = max(MDD_rate, DD_rate)
        return MDD_rate

    def GetAverEarn(self):
        if len(self.Profit) == 0:
            return 0
        WinProfit = [i for i in self.Profit if i > 0]
        return sum(WinProfit) / len(WinProfit) if len(WinProfit) > 0 else 0

    def GetAverLoss(self):
        if len(self.Profit) == 0:
            return 0
        FailProfit = [i for i in self.Profit if i < 0]
        return sum(FailProfit) / len(FailProfit) if len(FailProfit) > 0 else 0

    def GetCumulativeProfit(self):
        TotalProfit = [0]
        for i in self.Profit:
            TotalProfit.append(TotalProfit[-1] + i)
        return TotalProfit

    def GetCumulativeProfit_rate(self):
        TotalProfit_rate = [0]
        for i in self.Profit_rate:
            TotalProfit_rate.append(TotalProfit_rate[-1] + i)
        return TotalProfit_rate

    # ----------------------------
    # 繪圖
    # ----------------------------
    def GeneratorProfitChart(self, StrategyName='Strategy'):
        matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
        matplotlib.rcParams['axes.unicode_minus'] = False

        plt.figure()

        TotalProfit = self.GetCumulativeProfit()
        plt.plot(TotalProfit[1:], '-', marker='o', linewidth=1)

        plt.title('累計盈虧(元)')
        plt.xlabel('交易編號')
        plt.ylabel('累計盈虧(元)')

        length = len(TotalProfit)
        new_ticks = range(1, length + 1)
        plt.xticks(ticks=range(length), labels=new_ticks)

        st.pyplot(plt)
        plt.close()

    def GeneratorProfit_rateChart(self, StrategyName='Strategy'):
        matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
        matplotlib.rcParams['axes.unicode_minus'] = False

        plt.figure()

        TotalProfit_rate = self.GetCumulativeProfit_rate()
        plt.plot(TotalProfit_rate[1:], '-', marker='o', linewidth=1)

        plt.title('累計投資報酬率')
        plt.xlabel('交易編號')
        plt.ylabel('累計投資報酬率')

        length = len(TotalProfit_rate)
        new_ticks = range(1, length + 1)
        plt.xticks(ticks=range(length), labels=new_ticks)

        st.pyplot(plt)
        plt.close()