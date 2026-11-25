"""
Windows DPI-aware screen & window capture utilities using Win32 GDI (pywin32 + ctypes).
Zero heavy external dependencies; clean resource cleanup on every capture.
"""
from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np
from PIL import Image, ImageGrab
import win32api
import win32con
import win32gui
import win32process
import win32ui


# Make process Per-Monitor DPI Aware v2 so Win32 coordinates match physical pixels 1:1 across monitors
def enable_dpi_awareness():
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


enable_dpi_awareness()

PW_RENDERFULLCONTENT = 2
SRCCOPY_BLT = win32con.SRCCOPY | getattr(win32con, 'CAPTUREBLT', 0x40000000)

_OWN_TITLE_KEYWORDS = ('雀魂屏幕识别', '何切分析助手', '识别结果标注预览')
_MAJSOUL_PROC_KEYWORDS = ('jantama', 'mahjongsoul', 'mahjong_soul')
_MAJSOUL_TITLE_KEYWORDS = ('雀魂', 'mahjongsoul', 'mahjong soul', 'jantama', 'じゃんたま')


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    rect: Tuple[int, int, int, int]  # client rect in screen coords: (left, top, width, height)
    is_majsoul: bool = False
    monitor_label: str = ''
    process_name: str = ''


def get_virtual_screen_rect() -> Tuple[int, int, int, int]:
    """Return full virtual desktop bounding rect (vx, vy, vw, vh) spanning all monitors."""
    user32 = ctypes.windll.user32
    vx = int(user32.GetSystemMetrics(76))  # SM_XVIRTUALSCREEN
    vy = int(user32.GetSystemMetrics(77))  # SM_YVIRTUALSCREEN
    vw = int(user32.GetSystemMetrics(78))  # SM_CXVIRTUALSCREEN
    vh = int(user32.GetSystemMetrics(79))  # SM_CYVIRTUALSCREEN
    if vw <= 0 or vh <= 0:
        vw = int(user32.GetSystemMetrics(0))
        vh = int(user32.GetSystemMetrics(1))
        return (0, 0, max(1, vw), max(1, vh))
    return (vx, vy, vw, vh)


def get_monitors() -> List[Tuple[int, int, int, int]]:
    """Return list of monitor rects (left, top, right, bottom) ordered with primary first."""
    mons: List[Tuple[int, int, int, int, bool]] = []
    try:
        for hmon, _hdc, rect in win32api.EnumDisplayMonitors():
            info = win32api.GetMonitorInfo(hmon)
            is_primary = bool(info.get('Flags', 0) & 1)
            mons.append((rect[0], rect[1], rect[2], rect[3], is_primary))
    except Exception:
        vx, vy, vw, vh = get_virtual_screen_rect()
        return [(vx, vy, vx + vw, vy + vh)]
    mons.sort(key=lambda m: (not m[4], m[0], m[1]))
    return [(m[0], m[1], m[2], m[3]) for m in mons]


def get_monitor_label_for_hwnd(hwnd: int) -> str:
    """Return short label like '屏1' or '屏2' for the monitor hosting `hwnd`."""
    try:
        mons = get_monitors()
        if len(mons) <= 1:
            return ''
        hmon = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST)
        info = win32api.GetMonitorInfo(hmon)
        mrect = tuple(info.get('Monitor', ()))
        for idx, r in enumerate(mons, start=1):
            if r == mrect:
                return f'屏{idx}'
    except Exception:
        pass
    return ''


def _get_process_basename(hwnd: int) -> str:
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        if not pid:
            return ''
        # PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        hproc = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not hproc:
            return ''
        try:
            buf = ctypes.create_unicode_buffer(512)
            sz = ctypes.c_ulong(len(buf))
            if ctypes.windll.kernel32.QueryFullProcessImageNameW(hproc, 0, buf, ctypes.byref(sz)):
                return os.path.basename(buf.value)
        finally:
            ctypes.windll.kernel32.CloseHandle(hproc)
    except Exception:
        pass
    return ''


def get_client_screen_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
    if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
        return None
    try:
        l, t, r, b = win32gui.GetClientRect(hwnd)
        w, h = r - l, b - t
        if w < 100 or h < 100:
            return None
        sx, sy = win32gui.ClientToScreen(hwnd, (0, 0))
        return (sx, sy, w, h)
    except Exception:
        return None


def list_candidate_windows(exclude_hwnds: Optional[Set[int]] = None) -> List[WindowInfo]:
    """List visible top-level windows across all monitors, prioritizing Mahjong Soul / browsers."""
    results: List[WindowInfo] = []
    excl = exclude_hwnds or set()

    def enum_cb(hwnd, _):
        try:
            if hwnd in excl:
                return True
            if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if not title or not title.strip():
                return True
            t_clean = title.strip()
            if any(k in t_clean for k in _OWN_TITLE_KEYWORDS):
                return True
            rect = get_client_screen_rect(hwnd)
            if rect is None or rect[2] < 320 or rect[3] < 240:
                return True
            proc = _get_process_basename(hwnd)
            pl = proc.lower()
            tl = t_clean.lower()
            is_mj_proc = any(k in pl for k in _MAJSOUL_PROC_KEYWORDS)
            is_mj_title = any(k in tl for k in _MAJSOUL_TITLE_KEYWORDS)
            mon_lbl = get_monitor_label_for_hwnd(hwnd)
            results.append(WindowInfo(
                hwnd=hwnd,
                title=t_clean,
                rect=rect,
                is_majsoul=is_mj_proc or is_mj_title,
                monitor_label=mon_lbl,
                process_name=proc,
            ))
        except Exception:
            pass
        return True

    try:
        win32gui.EnumWindows(enum_cb, None)
    except Exception:
        pass

    def priority_key(w: WindowInfo):
        pl = w.process_name.lower()
        tl = w.title.lower()
        area = w.rect[2] * w.rect[3]
        if any(k in pl for k in _MAJSOUL_PROC_KEYWORDS):
            return (0, -area)
        if any(k in tl for k in _MAJSOUL_TITLE_KEYWORDS):
            return (1, -area)
        if any(k in tl for k in ('chrome', 'edge', 'firefox', 'bilibili', 'steam')):
            return (2, -area)
        return (3, -area)

    results.sort(key=priority_key)
    return results


def find_majsoul_window(exclude_hwnds: Optional[Set[int]] = None) -> Optional[WindowInfo]:
    """Return the highest-priority Mahjong Soul window if currently open, else None."""
    for w in list_candidate_windows(exclude_hwnds=exclude_hwnds):
        if w.is_majsoul:
            return w
    return None


def capture_rect(left: int, top: int, width: int, height: int) -> Optional[Image.Image]:
    """Capture a virtual-screen rectangle via GetDC(0) BitBlt into an RGB PIL Image, with ImageGrab fallback."""
    if width <= 0 or height <= 0:
        return None
    screen_dc = win32gui.GetDC(0)
    img_dc = win32ui.CreateDCFromHandle(screen_dc)
    mem_dc = img_dc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    old_bmp = None
    try:
        bmp.CreateCompatibleBitmap(img_dc, width, height)
        old_bmp = mem_dc.SelectObject(bmp)
        try:
            mem_dc.BitBlt((0, 0), (width, height), img_dc, (left, top), SRCCOPY_BLT)
        except Exception:
            mem_dc.BitBlt((0, 0), (width, height), img_dc, (left, top), win32con.SRCCOPY)
        bmp_info = bmp.GetInfo()
        bmp_bits = bmp.GetBitmapBits(True)
        img = Image.frombuffer(
            'RGB',
            (bmp_info['bmWidth'], bmp_info['bmHeight']),
            bmp_bits,
            'raw',
            'BGRX',
            0,
            1,
        )
        if float(np.asarray(img).mean()) > 1.0:
            return img.copy()
        raise RuntimeError('Blank BitBlt frame')
    except Exception:
        try:
            return ImageGrab.grab(bbox=(left, top, left + width, top + height), all_screens=True).convert('RGB')
        except Exception:
            return None
    finally:
        if old_bmp is not None:
            try:
                mem_dc.SelectObject(old_bmp)
            except Exception:
                pass
        try:
            win32gui.DeleteObject(bmp.GetHandle())
            mem_dc.DeleteDC()
            img_dc.DeleteDC()
            win32gui.ReleaseDC(0, screen_dc)
        except Exception:
            pass


def capture_window_client(hwnd: int, use_print_window: bool = False) -> Optional[Image.Image]:
    """Capture the client area of a window. Uses screen BitBlt by default for DX11 reliability."""
    rect = get_client_screen_rect(hwnd)
    if rect is None:
        return None
    left, top, w, h = rect

    if use_print_window:
        try:
            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            bmp = win32ui.CreateBitmap()
            wl, wt, wr, wb = win32gui.GetWindowRect(hwnd)
            ww, wh = wr - wl, wb - wt
            bmp.CreateCompatibleBitmap(mfc_dc, ww, wh)
            old_bmp = save_dc.SelectObject(bmp)
            try:
                ok = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)
                if ok:
                    bits = bmp.GetBitmapBits(True)
                    full = Image.frombuffer('RGB', (ww, wh), bits, 'raw', 'BGRX', 0, 1)
                    cx, cy = left - wl, top - wt
                    crop = full.crop((cx, cy, cx + w, cy + h))
                    if np.asarray(crop).mean() > 2.5:
                        return crop.copy()
            finally:
                try:
                    save_dc.SelectObject(old_bmp)
                except Exception:
                    pass
                win32gui.DeleteObject(bmp.GetHandle())
                save_dc.DeleteDC()
                mfc_dc.DeleteDC()
                win32gui.ReleaseDC(hwnd, hwnd_dc)
        except Exception:
            pass

    return capture_rect(left, top, w, h)


def capture_window_strip(
    hwnd: int,
    y_frac_start: float = 0.76,
    use_print_window: bool = False,
) -> Optional[Image.Image]:
    """Capture only the bottom hand strip of a window for fast idle monitor polling."""
    rect = get_client_screen_rect(hwnd)
    if rect is None:
        return None
    left, top, w, h = rect
    y0 = max(0, min(h - 1, int(h * y_frac_start)))
    sh = h - y0
    if sh < 30:
        return None
    if not use_print_window:
        return capture_rect(left, top + y0, w, sh)
    img = capture_window_client(hwnd, use_print_window=True)
    if img is None:
        return None
    return img.crop((0, y0, w, h))


def capture_burst(
    capture_fn,
    count: int = 3,
    interval_sec: float = 0.12,
) -> List[np.ndarray]:
    """Capture `count` RGB uint8 numpy frames spaced by `interval_sec` for dora shimmer calibration."""
    frames: List[np.ndarray] = []
    for i in range(count):
        if i > 0:
            time.sleep(interval_sec)
        img = capture_fn()
        if img is None:
            break
        frames.append(np.asarray(img.convert('RGB'), dtype=np.uint8))
    return frames
