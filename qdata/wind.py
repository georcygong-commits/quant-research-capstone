# -*- coding: utf-8 -*-
"""Wind 连接与请求封装：单例、重连、限频、错误重试。"""
import sys
import time
import loguru
import config

logger = loguru.logger

_w = None


def get_w():
    """返回已连接的 WindPy.w 单例。"""
    global _w
    if _w is not None and _w.isconnected():
        return _w
    try:
        from WindPy import w
    except ImportError:
        sys.path.append(config.WIND_X64_DIR)
        from WindPy import w
    if not w.isconnected():
        r = w.start()
        if r.ErrorCode != 0:
            raise RuntimeError(f"Wind start 失败: code={r.ErrorCode}, msg={r.Data}")
    _w = w
    return _w


def _is_ok(resp):
    # ErrorCode == 0 表示成功
    return getattr(resp, "ErrorCode", -1) == 0


def request(fn_name, *args, retry=None, sleep=None, **kwargs):
    """
    统一请求封装，带重试与限频。
    fn_name: 'wsd' / 'wss' / 'wsi' / 'wset' / 'tdays'
    """
    w = get_w()
    fn = getattr(w, fn_name)
    retry = config.WIND_RETRY if retry is None else retry
    sleep = config.WIND_SLEEP if sleep is None else sleep
    last_err = None
    for attempt in range(retry):
        if not w.isconnected():
            get_w()  # 触发重连
            fn = getattr(get_w(), fn_name)
        try:
            resp = fn(*args, **kwargs)
        except Exception as e:  # 接口本身异常
            last_err = repr(e)
            time.sleep(sleep * (attempt + 1))
            continue
        if _is_ok(resp):
            time.sleep(sleep)
            return resp
        last_err = f"ErrorCode={resp.ErrorCode}, Data={getattr(resp, 'Data', None)}"
        # -4052x 多为参数/权限类错误，重试无意义；网络/限频类错误退避后重试
        time.sleep(sleep * (attempt + 1) * 2)
    raise RuntimeError(f"Wind.{fn_name} 多次失败: {last_err}; args={args}")


def close():
    global _w
    if _w is not None:
        try:
            _w.close()
        except Exception:
            pass
        _w = None
