# -*- coding: utf-8 -*-
"""
监控/日志/容灾/灰度（M6.8）：
指标采集、健康检查、灰度路由、备份恢复骨架（Prometheus/K8s 风格的本地可运行模拟）。
"""
import random


class MetricsRegistry:
    """指标注册表：计数器/仪表盘，可导出快照（对应 Prometheus 采集）。"""
    def __init__(self):
        self._metrics = {}

    def inc(self, name, by=1):
        self._metrics[name] = self._metrics.get(name, 0) + by

    def gauge(self, name, value):
        self._metrics[name] = value

    def snapshot(self):
        return dict(self._metrics)


class HealthCheck:
    """健康检查：注册探针，聚合存活/就绪状态（对应 K8s liveness/readiness）。"""
    def __init__(self):
        self._probes = {}

    def register(self, name, probe):
        self._probes[name] = probe

    def check(self):
        results = {n: bool(p()) for n, p in self._probes.items()}
        return {"healthy": all(results.values()), "probes": results}


class CanaryRouter:
    """灰度路由：按比例把流量在稳定版与灰度版之间分流，逐步放量。"""
    def __init__(self, canary_pct=0.0):
        self.canary_pct = canary_pct

    def route(self, rng=random):
        return "canary" if rng.random() < self.canary_pct else "stable"

    def rollout(self, pct):
        self.canary_pct = max(0.0, min(1.0, pct))
        return self.canary_pct


class BackupManager:
    """容灾备份：记录备份点，支持恢复校验。"""
    def __init__(self):
        self.backups = []

    def backup(self, label):
        self.backups.append(label)
        return label

    def restore(self, label):
        if label not in self.backups:
            return {"ok": False, "reason": "backup not found"}
        return {"ok": True, "restored": label}
