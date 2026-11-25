"""
雀魂麻将屏幕识别与何切辅助工具 (Single-file Standalone GUI Application)
- 随开随用，关闭干净退出
- 支持：原版主题 / 猫主题（含角标/无角标）、全部62种牌背、4方向+手牌投影、暗杠2牌背+2牌面识别、宝牌防闪烁连拍校准
- 输出：牌效最佳何切 / 防御最佳何切 / 牌效+防御最佳何切 + 打出后向听数与向听枚数
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageTk
import win32gui

from board_recognizer import BoardRecognizer, RecognitionResult, draw_annotations
from mahjong_engine import (
    BoardState,
    CandidateEval,
    RecommendationResult,
    code_to_34,
    code_to_cn,
    format_shanten,
    idx34_to_code,
    recommend_discards,
)
from screen_capture import (
    WindowInfo,
    capture_burst,
    capture_rect,
    capture_window_client,
    capture_window_strip,
    find_majsoul_window,
    get_client_screen_rect,
    get_virtual_screen_rect,
    list_candidate_windows,
)

# Color palette (clean dark slate theme)
BG_MAIN = '#161920'
BG_CARD = '#1F242E'
BG_CARD_ALT = '#262C38'
BG_HEADER = '#11141A'
FG_TEXT = '#ECEFF4'
FG_MUTED = '#9AA4B2'
ACCENT_EFF = '#38BDF8'   # cyan-blue
ACCENT_DEF = '#34D399'   # emerald-green
ACCENT_BAL = '#FBBF24'   # amber-gold
ACCENT_WARN = '#F87171'  # coral-red
BORDER_CLR = '#303846'


def parse_compact_tiles(text: str) -> List[str]:
    """Parse compact notation like '123m406p3478s55z5m9p' or '1m 2m ...' into tile codes."""
    text = text.strip().lower().replace(',', ' ').replace('，', ' ')
    out: List[str] = []
    # Match digit sequences followed by m/p/s/z
    for digits, suit in re.findall(r'([0-9]+)([mpsz])', text):
        for ch in digits:
            if suit == 'z' and ch not in '1234567':
                continue
            if suit in ('m', 'p', 's') and ch not in '0123456789':
                continue
            out.append(ch + suit)
    return out


def format_compact_tiles(tiles: List[str]) -> str:
    return ''.join(tiles)


class RegionSelectorOverlay(tk.Toplevel):
    """Transparent multi-monitor drag-to-select ROI overlay."""

    def __init__(self, master, on_selected):
        super().__init__(master)
        self.on_selected = on_selected
        self.overrideredirect(True)
        vx, vy, vw, vh = get_virtual_screen_rect()
        self.vx = vx
        self.vy = vy
        self.geometry(f'{vw}x{vh}{vx:+d}{vy:+d}')
        self.attributes('-topmost', True)
        self.attributes('-alpha', 0.28)
        self.configure(bg='#000000', cursor='crosshair')

        self.canvas = tk.Canvas(self, bg='#000000', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        pw = self.winfo_screenwidth()
        self.canvas.create_text(
            pw // 2 - vx, 48 - vy,
            text='请按住鼠标左键框选雀魂牌局画面区域（支持多屏幕，ESC 取消）',
            fill='#FFFFFF', font=('Microsoft YaHei UI', 16, 'bold')
        )

        self.start_x = 0
        self.start_y = 0
        self.rect_id = None

        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.bind('<Escape>', lambda _e: self._cancel())
        self.focus_force()

    def _on_press(self, event):
        self.start_x, self.start_y = event.x_root, event.y_root
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        cx, cy = self.start_x - self.vx, self.start_y - self.vy
        self.rect_id = self.canvas.create_rectangle(
            cx, cy, cx, cy,
            outline='#38BDF8', width=3, fill='#38BDF8'
        )

    def _on_drag(self, event):
        if self.rect_id:
            self.canvas.coords(
                self.rect_id,
                self.start_x - self.vx,
                self.start_y - self.vy,
                event.x_root - self.vx,
                event.y_root - self.vy,
            )

    def _on_release(self, event):
        x0 = min(self.start_x, event.x_root)
        y0 = min(self.start_y, event.y_root)
        x1 = max(self.start_x, event.x_root)
        y1 = max(self.start_y, event.y_root)
        self.destroy()
        if (x1 - x0) >= 200 and (y1 - y0) >= 120:
            self.on_selected((x0, y0, x1 - x0, y1 - y0))

    def _cancel(self):
        self.destroy()


class MajsoulAdvisorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('雀魂屏幕识别 & 何切分析助手')
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        win_w = min(1120, max(940, sw - 60))
        win_h = min(800, max(660, sh - 80))
        self.root.geometry(f'{win_w}x{win_h}')
        self.root.minsize(920, 640)
        self.root.configure(bg=BG_MAIN)

        self.stop_event = threading.Event()
        self.worker_lock = threading.Lock()
        self._cfg_lock = threading.Lock()
        self.busy = False

        self.recognizer = BoardRecognizer()
        self.windows: List[WindowInfo] = []
        self.custom_rect: Optional[Tuple[int, int, int, int]] = None
        self.user_pinned_target = False
        self.auto_waiting_majsoul = True
        self.last_capture_img: Optional[Image.Image] = None
        self.last_rec: Optional[RecognitionResult] = None
        self.last_board: Optional[BoardState] = None

        # Thread-safe snapshots for the background continuous hand monitor
        self._cfg_monitor_enabled = True
        self._cfg_theme_pref = 'auto'
        self._cfg_mode_pref = 'auto'
        self._cfg_burst = True
        self._cfg_printwin = False
        self._cfg_custom_rect: Optional[Tuple[int, int, int, int]] = None
        self._cfg_target_hwnd: Optional[int] = None
        self._cfg_target_label = ''
        self._cfg_target_is_majsoul = False
        self._cfg_user_pinned = False
        self._cfg_auto_waiting = True

        self._last_analyzed_sig: Optional[Tuple[str, ...]] = None
        self._monitor_was_waiting = True
        self._reset_monitor_strip = False

        # Pre-render Tk PhotoImage icons for all 37 tiles
        self.tk_icons: Dict[str, ImageTk.PhotoImage] = {}
        self.tk_icons_sm: Dict[str, ImageTk.PhotoImage] = {}
        for code, pil_im in self.recognizer.ui_icons.items():
            self.tk_icons[code] = ImageTk.PhotoImage(pil_im.resize((44, 70), Image.Resampling.LANCZOS))
            self.tk_icons_sm[code] = ImageTk.PhotoImage(pil_im.resize((28, 45), Image.Resampling.LANCZOS))

        self._setup_styles()
        self._build_ui()
        self.root.update_idletasks()
        self._refresh_windows(auto_attach=True)
        self._sync_cfg_snapshot()
        self._start_monitor_thread()

        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass
        style.configure('TFrame', background=BG_MAIN)
        style.configure('Card.TFrame', background=BG_CARD)
        style.configure('TLabel', background=BG_MAIN, foreground=FG_TEXT, font=('Microsoft YaHei UI', 10))
        style.configure('Muted.TLabel', background=BG_MAIN, foreground=FG_MUTED, font=('Microsoft YaHei UI', 9))
        style.configure('TCheckbutton', background=BG_HEADER, foreground=FG_TEXT, font=('Microsoft YaHei UI', 9))
        style.configure('TCombobox', fieldbackground=BG_CARD_ALT, background=BG_CARD_ALT, foreground=FG_TEXT)
        style.map(
            'TCombobox',
            fieldbackground=[('readonly', BG_CARD_ALT)],
            foreground=[('readonly', FG_TEXT)],
            selectbackground=[('readonly', BG_CARD_ALT)],
            selectforeground=[('readonly', FG_TEXT)],
        )
        style.configure(
            'Treeview',
            background=BG_CARD,
            fieldbackground=BG_CARD,
            foreground=FG_TEXT,
            rowheight=26,
            borderwidth=0,
            font=('Microsoft YaHei UI', 9),
        )
        style.configure(
            'Treeview.Heading',
            background=BG_CARD_ALT,
            foreground=FG_TEXT,
            font=('Microsoft YaHei UI', 9, 'bold'),
        )
        style.map('Treeview', background=[('selected', '#334155')])

    def _build_ui(self):
        # Top control bar
        top_bar = tk.Frame(self.root, bg=BG_HEADER, padx=12, pady=8)
        top_bar.pack(fill=tk.X)

        tk.Label(top_bar, text='目标窗口:', bg=BG_HEADER, fg=FG_TEXT, font=('Microsoft YaHei UI', 9, 'bold')).pack(side=tk.LEFT)
        self.win_var = tk.StringVar()
        self.win_combo = ttk.Combobox(top_bar, textvariable=self.win_var, state='readonly', width=28)
        self.win_combo.pack(side=tk.LEFT, padx=(6, 4))
        self.win_combo.bind('<<ComboboxSelected>>', self._on_window_selected)

        self._btn(top_bar, '刷新', self._on_click_refresh, bg=BG_CARD_ALT).pack(side=tk.LEFT, padx=2)
        self._btn(top_bar, '框选区域', self._start_region_select, bg=BG_CARD_ALT).pack(side=tk.LEFT, padx=2)
        self._btn(top_bar, '读取截图', self._load_image_file, bg=BG_CARD_ALT).pack(side=tk.LEFT, padx=2)

        tk.Label(top_bar, text='主题:', bg=BG_HEADER, fg=FG_MUTED, font=('Microsoft YaHei UI', 9)).pack(side=tk.LEFT, padx=(8, 2))
        self.theme_var = tk.StringVar(value='自动')
        theme_cb = ttk.Combobox(top_bar, textvariable=self.theme_var, state='readonly', width=6, values=['自动', '原版', '猫主题'])
        theme_cb.pack(side=tk.LEFT, padx=2)
        theme_cb.bind('<<ComboboxSelected>>', lambda _e: self._sync_cfg_snapshot())

        tk.Label(top_bar, text='模式:', bg=BG_HEADER, fg=FG_MUTED, font=('Microsoft YaHei UI', 9)).pack(side=tk.LEFT, padx=(6, 2))
        self.mode_var = tk.StringVar(value='自动')
        mode_cb = ttk.Combobox(top_bar, textvariable=self.mode_var, state='readonly', width=5, values=['自动', '四麻', '三麻'])
        mode_cb.pack(side=tk.LEFT, padx=2)
        mode_cb.bind('<<ComboboxSelected>>', lambda _e: self._sync_cfg_snapshot())

        self.burst_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top_bar, text='宝牌防闪烁(3帧)', variable=self.burst_var, command=self._sync_cfg_snapshot).pack(side=tk.LEFT, padx=(8, 2))

        self.printwin_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top_bar, text='后台穿透', variable=self.printwin_var, command=self._sync_cfg_snapshot).pack(side=tk.LEFT, padx=2)

        self.topmost_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top_bar, text='置顶', variable=self.topmost_var, command=self._toggle_topmost).pack(side=tk.LEFT, padx=2)

        # Action bar
        act_bar = tk.Frame(self.root, bg=BG_MAIN, padx=12, pady=6)
        act_bar.pack(fill=tk.X)

        self.btn_once = self._btn(act_bar, '▶ 立即识别', self.trigger_recognize_once, bg='#0284C7', fg='#FFFFFF', bold=True)
        self.btn_once.pack(side=tk.LEFT, padx=(0, 6))

        self.monitor_var = tk.BooleanVar(value=True)
        self.btn_monitor = tk.Checkbutton(
            act_bar, text='● 持续监控中（摸牌自动分析）', variable=self.monitor_var,
            command=self._toggle_monitor, bg='#0F766E', fg='#FFFFFF',
            selectcolor='#0F766E', activebackground='#115E59', activeforeground='#FFFFFF',
            font=('Microsoft YaHei UI', 9, 'bold'), padx=10, pady=4, bd=0, indicatoron=False, cursor='hand2'
        )
        self.btn_monitor.pack(side=tk.LEFT, padx=4)

        self._btn(act_bar, '查看识别标注图', self._show_annotated_preview, bg=BG_CARD_ALT).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value='就绪：正在自动寻找雀魂麻将窗口...')
        tk.Label(act_bar, textvariable=self.status_var, bg=BG_MAIN, fg=FG_MUTED, font=('Microsoft YaHei UI', 9)).pack(side=tk.RIGHT)

        # Recognized Board Summary + Quick Manual Edit Strip
        board_frame = tk.Frame(self.root, bg=BG_CARD, padx=12, pady=8, highlightbackground=BORDER_CLR, highlightthickness=1)
        board_frame.pack(fill=tk.X, padx=12, pady=(2, 8))

        hand_row = tk.Frame(board_frame, bg=BG_CARD)
        hand_row.pack(fill=tk.X)
        tk.Label(hand_row, text='自家手牌:', bg=BG_CARD, fg=FG_TEXT, font=('Microsoft YaHei UI', 9, 'bold')).pack(side=tk.LEFT)
        self.hand_icons_frame = tk.Frame(hand_row, bg=BG_CARD)
        self.hand_icons_frame.pack(side=tk.LEFT, padx=8)

        edit_row = tk.Frame(board_frame, bg=BG_CARD)
        edit_row.pack(fill=tk.X, pady=(6, 0))
        tk.Label(edit_row, text='手牌校对:', bg=BG_CARD, fg=FG_MUTED, font=('Microsoft YaHei UI', 9)).pack(side=tk.LEFT)
        self.hand_edit_var = tk.StringVar()
        tk.Entry(edit_row, textvariable=self.hand_edit_var, bg=BG_CARD_ALT, fg=FG_TEXT, insertbackground=FG_TEXT,
                 relief=tk.FLAT, width=32, font=('Consolas', 10)).pack(side=tk.LEFT, padx=(4, 10))

        tk.Label(edit_row, text='宝牌指示:', bg=BG_CARD, fg=FG_MUTED, font=('Microsoft YaHei UI', 9)).pack(side=tk.LEFT)
        self.dora_edit_var = tk.StringVar()
        tk.Entry(edit_row, textvariable=self.dora_edit_var, bg=BG_CARD_ALT, fg=FG_TEXT, insertbackground=FG_TEXT,
                 relief=tk.FLAT, width=12, font=('Consolas', 10)).pack(side=tk.LEFT, padx=(4, 10))

        self._btn(edit_row, '按校对重算', self._recalc_from_manual, bg='#334155').pack(side=tk.LEFT, padx=2)
        self.meta_var = tk.StringVar(value='宝牌: - | 副露/立直: -')
        tk.Label(edit_row, textvariable=self.meta_var, bg=BG_CARD, fg=ACCENT_BAL, font=('Microsoft YaHei UI', 9)).pack(side=tk.RIGHT)

        # 6 Main Recommendation Hero Cards (2 rows x 3 cols)
        cards_row = tk.Frame(self.root, bg=BG_MAIN)
        cards_row.pack(fill=tk.X, padx=12, pady=(0, 6))
        for i in range(3):
            cards_row.columnconfigure(i, weight=1, uniform='rec_card')

        self.card_eff = self._create_rec_card(cards_row, 0, 0, '⚡ 牌效最佳何切', ACCENT_EFF)
        self.card_def = self._create_rec_card(cards_row, 0, 1, '🛡 防御最佳何切', ACCENT_DEF)
        self.card_bal = self._create_rec_card(cards_row, 0, 2, '⚖ 牌效+防御何切', ACCENT_BAL)

        self.card_pts = self._create_rec_card(cards_row, 1, 0, '💰 最佳点数何切', '#FBBF24')
        self.card_eff_pts = self._create_rec_card(cards_row, 1, 1, '🎯 牌效+点数何切', '#FB923C')
        self.card_def_pts = self._create_rec_card(cards_row, 1, 2, '🏰 防御+点数何切', '#2DD4BF')

        # Conditional Riichi Recommendation Row (shown only when self has riichi opportunity)
        self.riichi_frame = tk.Frame(self.root, bg='#2A1215', padx=8, pady=6,
                                     highlightbackground='#F43F5E', highlightthickness=1)
        tk.Label(self.riichi_frame, text='🔥 检测到自家有立直机会！以下为立直何切建议：',
                 bg='#2A1215', fg='#FDA4AF', font=('Microsoft YaHei UI', 9, 'bold'), anchor='w').pack(fill=tk.X, pady=(0, 4))
        riichi_grid = tk.Frame(self.riichi_frame, bg='#2A1215')
        riichi_grid.pack(fill=tk.X)
        for i in range(3):
            riichi_grid.columnconfigure(i, weight=1, uniform='riichi_card')
        self.card_r_eff = self._create_rec_card(riichi_grid, 0, 0, '🔥 最佳牌效立直何切', '#F43F5E')
        self.card_r_def = self._create_rec_card(riichi_grid, 0, 1, '🛡 最佳防御立直何切', '#34D399')
        self.card_r_pts = self._create_rec_card(riichi_grid, 0, 2, '💰 最佳点数立直何切', '#FBBF24')

        # Bottom Split: Left = All Candidates Table, Right = 4-Player Board Details
        self.bottom_frame = tk.Frame(self.root, bg=BG_MAIN)
        self.bottom_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))
        bottom = self.bottom_frame
        bottom.columnconfigure(0, weight=3)
        bottom.columnconfigure(1, weight=2)
        bottom.rowconfigure(0, weight=1)

        # Left: Candidate Table
        tbl_frame = tk.Frame(bottom, bg=BG_CARD, highlightbackground=BORDER_CLR, highlightthickness=1)
        tbl_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 6))
        tk.Label(tbl_frame, text='全部候选切牌对比（按综合评分排序）', bg=BG_CARD, fg=FG_TEXT,
                 font=('Microsoft YaHei UI', 9, 'bold'), anchor='w', padx=8, pady=5).pack(fill=tk.X)

        cols = ('tile', 'shanten', 'ukeire', 'points', 'danger', 'bal', 'tags')
        self.tree = ttk.Treeview(tbl_frame, columns=cols, show='headings', height=9)
        self.tree.heading('tile', text='切牌')
        self.tree.heading('shanten', text='切后向听')
        self.tree.heading('ukeire', text='进张')
        self.tree.heading('points', text='预估点数')
        self.tree.heading('danger', text='危险度')
        self.tree.heading('bal', text='综合分')
        self.tree.heading('tags', text='防御/牌型说明')

        self.tree.column('tile', width=72, anchor='center')
        self.tree.column('shanten', width=96, anchor='center')
        self.tree.column('ukeire', width=64, anchor='center')
        self.tree.column('points', width=72, anchor='center')
        self.tree.column('danger', width=60, anchor='center')
        self.tree.column('bal', width=60, anchor='center')
        self.tree.column('tags', width=180, anchor='w')
        self.tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        # Right: Recognized 4-Player Table State Log
        info_frame = tk.Frame(bottom, bg=BG_CARD, highlightbackground=BORDER_CLR, highlightthickness=1)
        info_frame.grid(row=0, column=1, sticky='nsew', padx=(6, 0))
        tk.Label(info_frame, text='牌局识别详情（牌河 / 副露 / 暗杠 / 立直）', bg=BG_CARD, fg=FG_TEXT,
                 font=('Microsoft YaHei UI', 9, 'bold'), anchor='w', padx=8, pady=5).pack(fill=tk.X)
        self.detail_text = tk.Text(
            info_frame, bg=BG_CARD, fg=FG_TEXT, relief=tk.FLAT, wrap=tk.WORD,
            font=('Microsoft YaHei UI', 9), padx=8, pady=4, state=tk.DISABLED
        )
        self.detail_text.pack(fill=tk.BOTH, expand=True)

    def _btn(self, parent, text, cmd, bg=BG_CARD_ALT, fg=FG_TEXT, bold=False):
        font = ('Microsoft YaHei UI', 9, 'bold' if bold else 'normal')
        return tk.Button(
            parent, text=text, command=cmd, bg=bg, fg=fg,
            activebackground='#475569', activeforeground='#FFFFFF',
            relief=tk.FLAT, padx=10, pady=4, font=font, cursor='hand2'
        )

    def _create_rec_card(self, parent, row_idx: int, col_idx: int, title: str, accent: str) -> dict:
        card = tk.Frame(parent, bg=BG_CARD, padx=8, pady=5, highlightbackground=accent, highlightthickness=1)
        padx = (0, 5) if col_idx == 0 else ((5, 0) if col_idx == 2 else (3, 3))
        pady = (0, 5) if row_idx == 0 else (0, 0)
        card.grid(row=row_idx, column=col_idx, sticky='nsew', padx=padx, pady=pady)

        tk.Label(card, text=title, bg=BG_CARD, fg=accent, font=('Microsoft YaHei UI', 9, 'bold')).pack(anchor='w')

        body = tk.Frame(card, bg=BG_CARD)
        body.pack(fill=tk.X, pady=(3, 2))

        icon_lbl = tk.Label(body, bg=BG_CARD)
        icon_lbl.pack(side=tk.LEFT, padx=(0, 8))

        right = tk.Frame(body, bg=BG_CARD)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tile_lbl = tk.Label(right, text='--', bg=BG_CARD, fg=FG_TEXT, font=('Microsoft YaHei UI', 13, 'bold'), anchor='w')
        tile_lbl.pack(fill=tk.X)

        sh_lbl = tk.Label(right, text='向听: --', bg=BG_CARD, fg=FG_TEXT, font=('Microsoft YaHei UI', 9), anchor='w')
        sh_lbl.pack(fill=tk.X)

        uk_lbl = tk.Label(right, text='进张: -- | 点数: --', bg=BG_CARD, fg=accent, font=('Microsoft YaHei UI', 9, 'bold'), anchor='w')
        uk_lbl.pack(fill=tk.X)

        sub_lbl = tk.Label(card, text='等待识别...', bg=BG_CARD, fg=FG_MUTED, font=('Microsoft YaHei UI', 8),
                           anchor='w', justify=tk.LEFT, wraplength=300)
        sub_lbl.pack(fill=tk.X, pady=(1, 0))

        return {'icon': icon_lbl, 'tile': tile_lbl, 'sh': sh_lbl, 'uk': uk_lbl, 'sub': sub_lbl}

    def _toggle_topmost(self):
        self.root.attributes('-topmost', self.topmost_var.get())

    def _own_hwnds(self) -> set:
        hwnds = set()
        try:
            wid = self.root.winfo_id()
            if wid:
                hwnds.add(wid)
                parent = win32gui.GetParent(wid)
                if parent:
                    hwnds.add(parent)
        except Exception:
            pass
        return hwnds

    @staticmethod
    def _format_window_label(w: WindowInfo) -> str:
        prefix = f'[{w.monitor_label}] ' if w.monitor_label else ''
        return f'{prefix}{w.title} ({w.rect[2]}x{w.rect[3]})'

    def _theme_pref_code(self) -> str:
        return {'自动': 'auto', '原版': 'default', '猫主题': 'cat'}.get(self.theme_var.get(), 'auto')

    def _mode_pref_code(self) -> str:
        return {'自动': 'auto', '四麻': '4p', '三麻': '3p'}.get(self.mode_var.get(), 'auto')

    def _sync_cfg_snapshot(self):
        idx = self.win_combo.current()
        win = self.windows[idx] if 0 <= idx < len(self.windows) else None
        with self._cfg_lock:
            self._cfg_monitor_enabled = bool(self.monitor_var.get())
            self._cfg_theme_pref = self._theme_pref_code()
            self._cfg_mode_pref = self._mode_pref_code()
            self._cfg_burst = bool(self.burst_var.get())
            self._cfg_printwin = bool(self.printwin_var.get())
            self._cfg_custom_rect = self.custom_rect
            self._cfg_user_pinned = self.user_pinned_target
            self._cfg_auto_waiting = self.auto_waiting_majsoul
            if win is not None:
                self._cfg_target_hwnd = win.hwnd
                self._cfg_target_label = win.title[:24]
                self._cfg_target_is_majsoul = win.is_majsoul
            else:
                self._cfg_target_hwnd = None
                self._cfg_target_label = ''
                self._cfg_target_is_majsoul = False

    def _on_click_refresh(self):
        self.user_pinned_target = False
        self.custom_rect = None
        self._refresh_windows(auto_attach=True)
        self._sync_cfg_snapshot()

    def _on_window_selected(self, _event=None):
        self.custom_rect = None
        self.auto_waiting_majsoul = False
        self.user_pinned_target = True
        self._reset_monitor_strip = True
        self._last_analyzed_sig = None
        self._monitor_was_waiting = True
        idx = self.win_combo.current()
        if 0 <= idx < len(self.windows):
            w = self.windows[idx]
            if w.is_majsoul:
                self.user_pinned_target = False
            self.status_var.set(f'已选择目标窗口: {w.title[:28]}')
        self._sync_cfg_snapshot()

    def _refresh_windows(self, auto_attach: bool = True):
        prev_hwnd = None
        cur_idx = self.win_combo.current()
        if 0 <= cur_idx < len(self.windows):
            prev_hwnd = self.windows[cur_idx].hwnd

        self.windows = list_candidate_windows(exclude_hwnds=self._own_hwnds())
        labels = [self._format_window_label(w) for w in self.windows]
        self.win_combo['values'] = labels

        if not self.windows:
            self.win_var.set('')
            self.auto_waiting_majsoul = (self.custom_rect is None and not self.user_pinned_target)
            self.status_var.set('等待自动连接雀魂窗口...（可先打开雀魂或点击「框选区域」）')
            self._sync_cfg_snapshot()
            return

        selected_idx = 0
        if auto_attach and not self.user_pinned_target:
            mj_indices = [i for i, w in enumerate(self.windows) if w.is_majsoul]
            if mj_indices:
                selected_idx = mj_indices[0]
                self.custom_rect = None
                self.auto_waiting_majsoul = False
                self.win_combo.current(selected_idx)
                mj_w = self.windows[selected_idx]
                self.status_var.set(f'已自动选中雀魂窗口: {mj_w.title} ({mj_w.rect[2]}x{mj_w.rect[3]})')
            else:
                self.win_combo.current(0)
                self.auto_waiting_majsoul = (self.custom_rect is None)
                if self.auto_waiting_majsoul:
                    self.status_var.set('未检测到雀魂窗口，等待自动连接（或手动选择窗口/框选区域）')
        else:
            if prev_hwnd is not None:
                for i, w in enumerate(self.windows):
                    if w.hwnd == prev_hwnd:
                        selected_idx = i
                        break
            self.win_combo.current(selected_idx)
            self.auto_waiting_majsoul = False

        self._sync_cfg_snapshot()

    def _auto_attach_window(self, mj_win: WindowInfo):
        if self.stop_event.is_set() or self.custom_rect is not None or self.user_pinned_target:
            return
        self._refresh_windows(auto_attach=True)
        self._reset_monitor_strip = True
        self._last_analyzed_sig = None
        self._monitor_was_waiting = True

    def _handle_target_lost(self):
        if self.stop_event.is_set():
            return
        self.auto_waiting_majsoul = True
        self.user_pinned_target = False
        self._reset_monitor_strip = True
        self._last_analyzed_sig = None
        self._monitor_was_waiting = True
        self._refresh_windows(auto_attach=True)

    def _start_region_select(self):
        RegionSelectorOverlay(self.root, self._on_region_selected)

    def _on_region_selected(self, rect: Tuple[int, int, int, int]):
        self.custom_rect = rect
        self.auto_waiting_majsoul = False
        self.user_pinned_target = True
        self._reset_monitor_strip = True
        self._last_analyzed_sig = None
        self._monitor_was_waiting = True
        self._sync_cfg_snapshot()
        self.status_var.set(f'已框选自定义区域: {rect[2]}x{rect[3]} @ ({rect[0]}, {rect[1]})')
        self.trigger_recognize_once()

    def _load_image_file(self):
        fp = filedialog.askopenfilename(
            title='选择雀魂截图文件',
            filetypes=[('图片文件', '*.png;*.jpg;*.jpeg;*.bmp;*.webp'), ('所有文件', '*.*')]
        )
        if not fp:
            return
        try:
            img = Image.open(fp).convert('RGB')
        except Exception as e:
            messagebox.showerror('打开失败', f'无法读取图片: {e}')
            return
        arr = np.asarray(img, dtype=np.uint8)
        self.last_capture_img = img
        self._run_recognition_async(static_frames=[arr], source_label=os.path.basename(fp))

    def _get_capture_fn(self):
        if self.custom_rect is not None:
            l, t, w, h = self.custom_rect
            return lambda: capture_rect(l, t, w, h), f'区域 {w}x{h}'
        idx = self.win_combo.current()
        if 0 <= idx < len(self.windows):
            win = self.windows[idx]
            use_pw = self.printwin_var.get()
            return lambda: capture_window_client(win.hwnd, use_print_window=use_pw), win.title[:24]
        return None, ''

    def trigger_recognize_once(self):
        if self.busy:
            return
        cap_fn, src_lbl = self._get_capture_fn()
        if cap_fn is None:
            self._refresh_windows(auto_attach=True)
            cap_fn, src_lbl = self._get_capture_fn()
        if cap_fn is None:
            self.status_var.set('未找到可用窗口，请点击「刷新」或「框选区域」')
            return
        self.auto_waiting_majsoul = False
        self._sync_cfg_snapshot()
        self._run_recognition_async(cap_fn=cap_fn, source_label=src_lbl)

    def _run_recognition_async(self, cap_fn=None, static_frames=None, source_label=''):
        with self.worker_lock:
            if self.busy:
                return
            self.busy = True
        self.status_var.set(f'正在识别 ({source_label})...')
        theme_pref = self._theme_pref_code()
        mode_pref = self._mode_pref_code()
        burst_n = 3 if self.burst_var.get() else 1

        def work():
            try:
                t0 = time.perf_counter()
                if static_frames is not None:
                    frames = static_frames
                else:
                    frames = capture_burst(cap_fn, count=burst_n, interval_sec=0.08)
                    if not frames:
                        self.root.after(0, lambda: self._on_worker_error('窗口截图失败（窗口可能已最小化或关闭）'))
                        return
                    self.last_capture_img = Image.fromarray(frames[-1])

                rec = self.recognizer.recognize_frames(frames, theme_pref=theme_pref, mode_pref=mode_pref)
                advice = recommend_discards(rec.board)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if not self.stop_event.is_set():
                    self.root.after(0, lambda: self._on_worker_done(rec, advice, elapsed_ms, source_label))
            except Exception as exc:
                if not self.stop_event.is_set():
                    self.root.after(0, lambda: self._on_worker_error(f'识别异常: {exc}'))

        threading.Thread(target=work, daemon=True).start()

    def _on_worker_error(self, msg: str):
        self.busy = False
        self.status_var.set(msg)

    def _set_monitor_status(self, msg: str):
        if not self.busy and not self.stop_event.is_set():
            self.status_var.set(msg)

    def _on_worker_done(self, rec: RecognitionResult, advice: RecommendationResult, elapsed_ms: float, src: str):
        self.busy = False
        self.last_rec = rec
        self.last_board = rec.board
        self._last_analyzed_sig = tuple(rec.board.hand) if rec.board.hand else None
        self._monitor_was_waiting = (len(rec.board.hand) % 3 != 2)
        theme_cn = '原版主题' if rec.detected_theme == 'default' else '猫主题'
        mode_cn = '三麻' if rec.board.player_count == 3 else '四麻'
        self.status_var.set(
            f'完成 [{src}] | {mode_cn} | {theme_cn} | 置信度 {rec.avg_confidence:.2f} | 耗时 {elapsed_ms:.0f}ms'
        )
        self._render_results(rec, advice)

    def _recalc_from_manual(self):
        hand = parse_compact_tiles(self.hand_edit_var.get())
        dora = parse_compact_tiles(self.dora_edit_var.get())
        if not hand:
            messagebox.showwarning('输入有误', '请输入有效的手牌，例如: 123m406p3478s55z5m9p')
            return
        mode_pref = self._mode_pref_code()
        if self.last_board is not None:
            p_cnt = 3 if mode_pref == '3p' else (4 if mode_pref == '4p' else self.last_board.player_count)
            board = BoardState(
                hand=hand,
                dora_indicators=dora,
                rivers=self.last_board.rivers,
                melds=self.last_board.melds,
                riichi=self.last_board.riichi,
                theme=self.last_board.theme,
                player_count=p_cnt,
                kita=self.last_board.kita if p_cnt == 3 else [0, 0, 0, 0],
            )
        else:
            p_cnt = 3 if mode_pref == '3p' else 4
            board = BoardState(hand=hand, dora_indicators=dora, player_count=p_cnt)
        self.last_board = board
        advice = recommend_discards(board)
        self._render_results(self.last_rec, advice, override_board=board)
        self.status_var.set(f'已按校对手牌 ({len(hand)}张) 重新计算')

    def _render_results(self, rec: Optional[RecognitionResult], advice: RecommendationResult, override_board: Optional[BoardState] = None):
        board = override_board or (rec.board if rec else BoardState(hand=[]))

        # 1. Self hand icons & edit fields
        for w in self.hand_icons_frame.winfo_children():
            w.destroy()
        for code in board.hand:
            ic = self.tk_icons_sm.get(code)
            if ic:
                tk.Label(self.hand_icons_frame, image=ic, bg=BG_CARD).pack(side=tk.LEFT, padx=1)

        self.hand_edit_var.set(format_compact_tiles(board.hand))
        self.dora_edit_var.set(format_compact_tiles(board.dora_indicators))

        dora_cn = ' '.join(code_to_cn(idx34_to_code(d)) for d in advice.dora_tiles34) or '未识别'
        mode_cn = '三麻' if board.player_count == 3 else '四麻'
        if board.player_count == 3:
            p_short = ['自家', '下家', '上家']
            riichi_players = [p_short[p] for p in (0, 1, 2) if board.riichi[p]]
            kita_str = f' | 拔北: 自{board.kita[0]}/下{board.kita[1]}/上{board.kita[2]}'
        else:
            p_short = ['自家', '下家', '对家', '上家']
            riichi_players = [p_short[p] for p in range(4) if board.riichi[p]]
            kita_str = ''
        r_txt = ','.join(riichi_players) if riichi_players else '无'
        self.meta_var.set(
            f'{mode_cn} | 当前向听: {format_shanten(advice.current_shanten)} | 宝牌: {dora_cn}{kita_str} | 立直: {r_txt}'
        )

        # 2. Update 6 main recommendation cards
        cur_sh = advice.current_shanten
        self._update_card(self.card_eff, advice.efficiency_best, 'eff', cur_sh, advice.summary_note)
        self._update_card(self.card_def, advice.defense_best, 'def', cur_sh, advice.summary_note)
        self._update_card(self.card_bal, advice.balanced_best, 'bal', cur_sh, advice.summary_note)
        self._update_card(self.card_pts, advice.points_best, 'pts', cur_sh, advice.summary_note)
        self._update_card(self.card_eff_pts, advice.eff_points_best, 'eff_pts', cur_sh, advice.summary_note)
        self._update_card(self.card_def_pts, advice.def_points_best, 'def_pts', cur_sh, advice.summary_note)

        # 2b. Show/hide temporary Riichi recommendation row
        if advice.can_riichi:
            if not self.riichi_frame.winfo_ismapped():
                self.riichi_frame.pack(fill=tk.X, padx=12, pady=(0, 6), before=self.bottom_frame)
            self._update_card(self.card_r_eff, advice.riichi_eff_best, 'r_eff', cur_sh, '')
            self._update_card(self.card_r_def, advice.riichi_def_best, 'r_def', cur_sh, '')
            self._update_card(self.card_r_pts, advice.riichi_points_best, 'r_pts', cur_sh, '')
        else:
            if self.riichi_frame.winfo_ismapped():
                self.riichi_frame.pack_forget()

        # 3. Populate candidate table
        for item in self.tree.get_children():
            self.tree.delete(item)
        for c in advice.candidates:
            delta = c.shanten - cur_sh
            delta_str = '(向听不变)' if delta == 0 else f'(向听{delta:+d})'
            sh_txt = f'{format_shanten(c.shanten)} {delta_str}'
            uk_txt = f'{c.ukeire}枚' + ('(振)' if c.furiten else '')
            pts_txt = f'{c.est_points}点'
            tags_txt = ' / '.join(c.danger_tags) if c.danger_tags else '-'
            self.tree.insert('', tk.END, values=(
                f'{code_to_cn(c.code)} ({c.code})',
                sh_txt,
                uk_txt,
                pts_txt,
                f'{c.danger:.1f}',
                f'{c.balanced_score:.1f}',
                tags_txt,
            ))

        # 4. Populate player detail log (3P or 4P)
        self.detail_text.configure(state=tk.NORMAL)
        self.detail_text.delete('1.0', tk.END)
        if board.player_count == 3:
            p_indices = (0, 1, 2)
            pnames = ['自家 (下)', '下家 (右)', '上家 (上)']
        else:
            p_indices = (0, 1, 2, 3)
            pnames = ['自家 (下)', '下家 (右)', '对家 (上)', '上家 (左)']
        lines = []
        for p in p_indices:
            rv_str = ' '.join(board.rivers[p]) if board.rivers[p] else '无'
            m_str = ' | '.join(' '.join(g) for g in board.melds[p]) if board.melds[p] else '无'
            r_tag = ' [立直!]' if board.riichi[p] else ''
            k_line = f'\n  拔北: {board.kita[p]} 枚' if board.player_count == 3 else ''
            lines.append(f'【{pnames[p]}】{r_tag}\n  牌河({len(board.rivers[p])}): {rv_str}\n  副露: {m_str}{k_line}')
        if rec and rec.notes:
            lines.append('\n提示:\n  ' + '\n  '.join(rec.notes))
        self.detail_text.insert('1.0', '\n'.join(lines))
        self.detail_text.configure(state=tk.DISABLED)

    def _update_card(self, card: dict, cand: Optional[CandidateEval], mode: str, cur_sh: int, fallback_note: str):
        if cand is None:
            card['icon'].configure(image='')
            card['tile'].configure(text='--')
            card['sh'].configure(text='向听: --')
            card['uk'].configure(text='进张: -- | 点数: --')
            card['sub'].configure(text=fallback_note or '需 3k+2 张手牌（摸牌待切状态）')
            return

        ic = self.tk_icons.get(cand.code)
        card['icon'].configure(image=ic if ic else '')
        card['tile'].configure(text=f'切 {code_to_cn(cand.code)} ({cand.code})')

        delta = cand.shanten - cur_sh
        delta_str = '(向听不变)' if delta == 0 else f'(向听{delta:+d})'
        card['sh'].configure(text=f'{format_shanten(cand.shanten)} {delta_str}')

        fur_str = ' [振听!]' if cand.furiten else ''
        card['uk'].configure(text=f'进张 {cand.ukeire}枚{fur_str} | 约{cand.est_points}点')

        # Format top improving tiles breakdown
        top_uk = sorted(cand.ukeire_tiles, key=lambda x: -x[1])[:6]
        uk_list_str = ' '.join(f'{code_to_cn(idx34_to_code(t))}×{rem}' for t, rem in top_uk)
        if len(cand.ukeire_tiles) > 6:
            uk_list_str += ' ...'
        tags_str = ' / '.join(cand.danger_tags) if cand.danger_tags else '常规'
        yaku_str = ' '.join(cand.yaku_tags[:3]) if cand.yaku_tags else '常规'

        if mode in ('eff', 'r_eff'):
            card['sub'].configure(text=f'进张: {uk_list_str or "无"} | 危险 {cand.danger:.1f} ({tags_str})')
        elif mode in ('def', 'r_def'):
            card['sub'].configure(text=f'危险 {cand.danger:.1f} ({tags_str}) | 进张: {uk_list_str or "无"}')
        elif mode in ('pts', 'r_pts'):
            card['sub'].configure(text=f'打点: {cand.est_han:.1f}番 ({yaku_str}) | 危险 {cand.danger:.1f}')
        elif mode == 'eff_pts':
            card['sub'].configure(text=f'打点: {cand.est_han:.1f}番 ({yaku_str}) | 进张: {uk_list_str or "无"}')
        elif mode == 'def_pts':
            card['sub'].configure(text=f'危险 {cand.danger:.1f} ({tags_str}) | 打点: {cand.est_han:.1f}番 ({yaku_str})')
        else:
            card['sub'].configure(text=f'综合 {cand.balanced_score:.1f} | 危险 {cand.danger:.1f} ({tags_str})')

    def _show_annotated_preview(self):
        if self.last_capture_img is None or self.last_rec is None:
            messagebox.showinfo('暂无截图', '请先进行一次屏幕识别或读取截图。')
            return
        ann = draw_annotations(self.last_capture_img, self.last_rec)
        # Fit within 1120x660
        ann.thumbnail((1120, 660), Image.Resampling.LANCZOS)
        win = tk.Toplevel(self.root)
        win.title('识别结果标注预览（绿=手牌 黄=宝牌指示 蓝=牌河 粉=副露/暗杠 紫=拔北宝牌）')
        win.configure(bg=BG_MAIN)
        photo = ImageTk.PhotoImage(ann)
        lbl = tk.Label(win, image=photo, bg=BG_MAIN)
        lbl.image = photo
        lbl.pack(padx=8, pady=8)

    def _toggle_monitor(self):
        enabled = bool(self.monitor_var.get())
        if enabled:
            self.btn_monitor.configure(
                text='● 持续监控中（摸牌自动分析）', bg='#0F766E', activebackground='#115E59'
            )
            self._reset_monitor_strip = True
        else:
            self.btn_monitor.configure(
                text='○ 开启持续监控（摸牌自动分析）', bg=BG_CARD_ALT, activebackground='#475569'
            )
        self._sync_cfg_snapshot()

    def _start_monitor_thread(self):
        threading.Thread(target=self._monitor_thread_fn, daemon=True).start()

    def _monitor_thread_fn(self):
        prev_tick_strip: Optional[np.ndarray] = None
        last_checked_strip: Optional[np.ndarray] = None
        pending_settle = False
        last_attach_check = 0.0
        cached_auto_theme: Optional[str] = None
        wait_sec = 0.15

        while not self.stop_event.wait(wait_sec):
            wait_sec = 0.15
            with self._cfg_lock:
                mon_enabled = self._cfg_monitor_enabled
                theme_pref = self._cfg_theme_pref
                mode_pref = self._cfg_mode_pref
                burst_on = self._cfg_burst
                use_pw = self._cfg_printwin
                custom_rect = self._cfg_custom_rect
                target_hwnd = self._cfg_target_hwnd
                target_label = self._cfg_target_label
                target_is_mj = self._cfg_target_is_majsoul
                user_pinned = self._cfg_user_pinned
                auto_waiting = self._cfg_auto_waiting

            if self._reset_monitor_strip:
                prev_tick_strip = None
                last_checked_strip = None
                pending_settle = False
                cached_auto_theme = None
                self._reset_monitor_strip = False

            if theme_pref != 'auto':
                cached_auto_theme = None

            if not mon_enabled or self.busy:
                continue

            now = time.monotonic()
            # Periodically auto-discover Majsoul window if not locked to custom rect/pinned window
            if custom_rect is None and (now - last_attach_check >= 2.0):
                last_attach_check = now
                if target_hwnd is not None and get_client_screen_rect(target_hwnd) is None:
                    self.root.after(0, self._handle_target_lost)
                    continue
                if (not user_pinned) and (not target_is_mj or target_hwnd is None):
                    mj = find_majsoul_window()
                    if mj is not None:
                        self.root.after(0, lambda w=mj: self._auto_attach_window(w))
                        continue

            if auto_waiting and custom_rect is None and not user_pinned:
                continue

            # Fast idle polling: capture only bottom 24% strip until motion/change is detected
            if not pending_settle and last_checked_strip is not None:
                if custom_rect is not None:
                    l, t, rw, rh = custom_rect
                    y0 = max(0, min(rh - 1, int(rh * 0.76)))
                    strip_img = capture_rect(l, t + y0, rw, rh - y0) if (rh - y0) >= 30 else None
                    src_lbl = f'区域 {rw}x{rh}'
                elif target_hwnd is not None:
                    strip_img = capture_window_strip(target_hwnd, y_frac_start=0.76, use_print_window=use_pw)
                    src_lbl = target_label or '雀魂麻将'
                else:
                    continue

                if strip_img is None:
                    continue
                cur_strip = np.asarray(
                    strip_img.convert('L').resize((160, 40), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
                diff_tick = float(np.mean(np.abs(cur_strip - prev_tick_strip))) if prev_tick_strip is not None else 99.0
                prev_tick_strip = cur_strip
                if diff_tick > 5.5:
                    pending_settle = True
                    wait_sec = 0.10
                    continue
                diff_checked = float(np.mean(np.abs(cur_strip - last_checked_strip)))
                if diff_checked < 2.2:
                    continue

            if custom_rect is not None:
                l, t, rw, rh = custom_rect
                img = capture_rect(l, t, rw, rh)
                src_lbl = f'区域 {rw}x{rh}'
            elif target_hwnd is not None:
                img = capture_window_client(target_hwnd, use_print_window=use_pw)
                src_lbl = target_label or '雀魂麻将'
            else:
                continue

            if img is None:
                continue

            t_arr = time.monotonic()
            arr = np.asarray(img.convert('RGB'), dtype=np.uint8)
            ah, _aw = arr.shape[:2]
            if ah < 120:
                continue
            strip = arr[int(ah * 0.76):ah, :]
            cur_strip = np.asarray(
                Image.fromarray(strip).convert('L').resize((160, 40), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )

            diff_tick = float(np.mean(np.abs(cur_strip - prev_tick_strip))) if prev_tick_strip is not None else 99.0
            prev_tick_strip = cur_strip
            if diff_tick > 5.5:
                pending_settle = True
                wait_sec = 0.10
                continue

            diff_checked = float(np.mean(np.abs(cur_strip - last_checked_strip))) if last_checked_strip is not None else 99.0
            if not pending_settle and last_checked_strip is not None and diff_checked < 2.2:
                continue
            pending_settle = False
            last_checked_strip = cur_strip

            try:
                quick_theme = cached_auto_theme if (theme_pref == 'auto' and cached_auto_theme) else theme_pref
                _th, hand_boxes = self.recognizer.recognize_hand_quick(arr, theme_pref=quick_theme)
                if theme_pref == 'auto' and cached_auto_theme is not None and len(hand_boxes) < 4:
                    cached_auto_theme = None
                    _th, hand_boxes = self.recognizer.recognize_hand_quick(arr, theme_pref='auto')
            except Exception:
                continue

            n = len(hand_boxes)
            if n >= 4:
                cached_auto_theme = _th
            sig = tuple(b.label for b in hand_boxes)
            if n == 0:
                cached_auto_theme = None
                self._monitor_was_waiting = True
                self._last_analyzed_sig = None
                self.root.after(0, lambda s=src_lbl: self._set_monitor_status(f'持续监控中 [{s}]：等待进入牌局...'))
                continue

            if n % 3 == 1:
                self._monitor_was_waiting = True
                self.root.after(
                    0,
                    lambda s=src_lbl, cnt=n: self._set_monitor_status(f'持续监控中 [{s}]：已识别 {cnt} 张手牌，等待摸牌...'),
                )
                continue

            if n % 3 == 2 and (self._monitor_was_waiting or sig != self._last_analyzed_sig):
                with self.worker_lock:
                    if self.busy:
                        continue
                    self.busy = True
                self.root.after(
                    0,
                    lambda s=src_lbl, cnt=n: self.status_var.set(f'检测到摸牌 ({cnt}张) [{s}]，正在分析何切...'),
                )
                try:
                    t0 = time.perf_counter()
                    frames = [arr]
                    if burst_on:
                        def _cap_once():
                            if custom_rect is not None:
                                return capture_rect(*custom_rect)
                            return capture_window_client(target_hwnd, use_print_window=use_pw)
                        rem_wait = 0.08 - (time.monotonic() - t_arr)
                        if rem_wait > 0.0:
                            self.stop_event.wait(rem_wait)
                        extra = capture_burst(_cap_once, count=2, interval_sec=0.08)
                        frames.extend(extra)

                    self.last_capture_img = Image.fromarray(frames[-1])
                    rec = self.recognizer.recognize_frames(
                        frames, theme_pref=theme_pref, mode_pref=mode_pref, hint_theme=cached_auto_theme
                    )
                    advice = recommend_discards(rec.board)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    if not self.stop_event.is_set():
                        self.root.after(0, lambda r=rec, a=advice, ms=elapsed_ms, s=src_lbl: self._on_worker_done(r, a, ms, s))
                except Exception as exc:
                    if not self.stop_event.is_set():
                        self.root.after(0, lambda e=exc: self._on_worker_error(f'自动分析异常: {e}'))

    def on_close(self):
        self.stop_event.set()
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass


def main():
    root = tk.Tk()
    app = MajsoulAdvisorApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
