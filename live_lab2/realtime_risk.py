# -*- coding: utf-8 -*-
"""
实时风控与熔断（M6.2）：
多层限额检查、熔断器状态机（NORMAL/WARNING/TRIPPED/RESET）、
kill switch 紧急停机。
"""
import numpy as np


class RiskLimits:
    """多层限额：单票权重、总杠杆、行业敞口、现金比例。"""
    def __init__(self, max_weight=0.05, max_gross=1.0, max_industry=0.3,
                 min_cash=0.05):
        self.limits = {"max_weight": max_weight, "max_gross": max_gross,
                       "max_industry": max_industry, "min_cash": min_cash}

    def check_order(self, new_weight, gross, industry_exp, cash_ratio):
        """下单前综合校验，返回 (是否放行, 违规列表)。"""
        violations = []
        if new_weight > self.limits["max_weight"]:
            violations.append("WEIGHT")
        if gross > self.limits["max_gross"]:
            violations.append("GROSS")
        if industry_exp > self.limits["max_industry"]:
            violations.append("INDUSTRY")
        if cash_ratio < self.limits["min_cash"]:
            violations.append("CASH")
        return (len(violations) == 0, violations)


class CircuitBreaker:
    """
    熔断器状态机：
      NORMAL（正常）→ 指标越限 TRIPPED（熔断，停止开新仓）
      → 人工/冷却后 RESET 回 NORMAL。
    """
    def __init__(self, threshold):
        self.threshold = threshold
        self.state = "NORMAL"

    def evaluate(self, metric):
        if self.state == "TRIPPED":
            return "TRIPPED"
        if metric >= self.threshold:
            self.state = "TRIPPED"
            return "TRIPPED"
        return "NORMAL"

    def reset(self):
        self.state = "NORMAL"
        return self.state


class KillSwitch:
    """紧急停机：触发后立即锁定、停止一切交易并撤单，需人工解锁。"""
    def __init__(self):
        self.armed = False
        self.reason = None

    def trigger(self, reason):
        self.armed = True
        self.reason = reason
        return {"kill": True, "cancel_all": True, "reason": reason}

    def disarm(self):
        self.armed = False
        self.reason = None
        return "DISARMED"
