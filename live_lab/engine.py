# -*- coding: utf-8 -*-
"""事件驱动引擎：事件注册、分发（实盘系统的核心骨架）。"""
from collections import defaultdict
from queue import Queue, Empty


class Event:
    def __init__(self, type, data=None):
        self.type = type
        self.data = data

    def __repr__(self):
        return "Event(%s)" % self.type


class EventEngine:
    """
    事件引擎：维护一个事件队列，按类型分发给注册的处理函数。
    实盘中由独立线程持续从队列取事件并处理。
    """

    def __init__(self):
        self._queue = Queue()
        self._handlers = defaultdict(list)
        self._general = []

    def register(self, type, handler):
        if type is None:
            self._general.append(handler)
        else:
            self._handlers[type].append(handler)

    def send(self, event):
        self._queue.put(event)

    def process_once(self, block=False, timeout=0):
        try:
            event = self._queue.get(block=block, timeout=timeout)
        except Empty:
            return False
        for h in self._handlers.get(event.type, []):
            h(event)
        for h in self._general:
            h(event)
        return True

    def run(self, max_events=None):
        count = 0
        while True:
            if not self.process_once(block=True, timeout=1):
                if self._queue.empty():
                    break
            count += 1
            if max_events and count >= max_events:
                break
