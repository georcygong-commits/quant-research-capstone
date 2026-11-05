# -*- coding: utf-8 -*-
"""实时风控：事前/事中/事后检查、限额、报警。"""
import numpy as np


class RiskManager:
    """
    实时风控器：维护一组限额规则，下单前/盘中/收盘后分别检查。
    不通过则拒绝并触发报警。
    """

    def __init__(self, limits):
        # limits 示例：{"max_weight":0.1, "min_cash":0.05, "max_drawdown":0.15}
        self.limits = limits
        self.alerts = []

    def alert(self, msg):
        self.alerts.append(msg)
        return msg

    def pre_trade_check(self, code, weight):
        """事前：单笔下单前校验权重上限。"""
        if weight > self.limits.get("max_weight", 1):
            return False, self.alert("拒绝：%s 权重 %.2f 超上限" % (code, weight))
        return True, "通过"

    def intraday_check(self, cash_ratio, drawdown):
        """事中：现金比例、实时回撤，触发则报警/熔断。"""
        if cash_ratio < self.limits.get("min_cash", 0):
            return False, self.alert("熔断：现金比例 %.2f 过低" % cash_ratio)
        if drawdown > self.limits.get("max_drawdown", 1):
            return False, self.alert("熔断：回撤 %.2f 超阈值" % drawdown)
        return True, "正常"

    def post_trade_report(self, nav, positions):
        """事后：汇总持仓集中度、净值，留档。"""
        return {"nav": nav, "n_positions": len(positions),
                "alerts": list(self.alerts)}
