# -*- coding: utf-8 -*-
"""
报警与人工接管（M6.4）：
分级报警（INFO/WARN/CRITICAL）与通知路由、
人工接管状态机（AUTO/REQUESTED/TAKEOVER/RETURN）与审计留痕。
"""
import numpy as np


LEVELS = {"INFO": 0, "WARN": 1, "CRITICAL": 2}


class AlertManager:
    """分级报警：按级别路由通知，记录全部报警并可统计。"""
    def __init__(self, routes=None):
        self.routes = routes or {"INFO": "log", "WARN": "im",
                                 "CRITICAL": "phone+im"}
        self.records = []

    def raise_alert(self, level, msg):
        if level not in LEVELS:
            raise ValueError("unknown level")
        rec = {"level": level, "msg": msg, "channel": self.routes[level]}
        self.records.append(rec)
        return rec

    def count(self, level):
        return sum(1 for r in self.records if r["level"] == level)


class HumanHandoff:
    """
    人工接管状态机：
      AUTO（自动运行）→ REQUESTED（请求接管）→ TAKEOVER（人工接管，暂停自动下单）
      → RETURN（人工交还，恢复 AUTO）。全程审计。
    """
    def __init__(self):
        self.state = "AUTO"
        self.operator = None
        self.audit = []

    def request(self, reason):
        self.state = "REQUESTED"
        self.audit.append({"action": "request", "reason": reason})
        return self.state

    def take_over(self, operator, authorized):
        if not authorized:
            self.audit.append({"action": "denied", "operator": operator})
            return "DENIED"
        self.state = "TAKEOVER"; self.operator = operator
        self.audit.append({"action": "takeover", "operator": operator})
        return self.state

    def return_to_auto(self, operator):
        self.state = "AUTO"; self.operator = None
        self.audit.append({"action": "return", "operator": operator})
        return self.state

    def auto_trading_blocked(self):
        return self.state == "TAKEOVER"
