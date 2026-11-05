# -*- coding: utf-8 -*-
"""
低延迟与并发系统设计（M6.6）：
线程安全共享状态、幂等消息处理、可注入时钟（回测/实盘一致性）。
"""
import threading
import time
from queue import Queue, Empty


class ThreadSafeCounter:
    """锁保护的共享计数器：多线程并发更新不丢更新。"""
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self._value += 1
            return self._value

    @property
    def value(self):
        with self._lock:
            return self._value


class IdempotentBus:
    """
    幂等消息总线：相同 idempotency_key 的消息只处理一次，
    重复投递（网络重试/故障重发）返回已处理结果而不重复执行。
    """
    def __init__(self):
        self._seen = set()
        self._results = {}
        self._lock = threading.Lock()

    def publish(self, key, handler):
        with self._lock:
            if key in self._seen:
                return {"dedup": True, "result": self._results[key]}
            result = handler()
            self._seen.add(key); self._results[key] = result
            return {"dedup": False, "result": result}


class Clock:
    """
    可注入时钟：回测用固定/回放时钟（可重复、可快进），
    实盘用系统时钟，策略只依赖 now()，保证回测与实盘代码一致。
    """
    def __init__(self, fixed=None):
        self._fixed = fixed

    def now(self):
        return self._fixed if self._fixed is not None else time.time()

    def advance(self, seconds):
        if self._fixed is not None:
            self._fixed += seconds


class MessageQueue:
    """线程间消息队列骨架：生产/消费解耦（实盘可用 Kafka/RabbitMQ 替换）。"""
    def __init__(self):
        self._q = Queue()

    def put(self, msg):
        self._q.put(msg)

    def get(self, timeout=0.1):
        try:
            return self._q.get(timeout=timeout)
        except Empty:
            return None
