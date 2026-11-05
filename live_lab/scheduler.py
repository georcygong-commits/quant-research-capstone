# -*- coding: utf-8 -*-
"""任务调度：把函数注册到特定时点/事件，按计划触发。"""


class Scheduler:
    """简单调度器：按时点（key）注册任务，run 时按到达的 key 触发。"""

    def __init__(self):
        self.tasks = {}

    def schedule(self, key, func, name=None):
        self.tasks.setdefault(key, []).append(
            (name or getattr(func, "__name__", "task"), func))

    def on_time(self, key, context=None):
        out = []
        for name, func in self.tasks.get(key, []):
            out.append((name, func(context)))
        return out

    def keys(self):
        return list(self.tasks)
