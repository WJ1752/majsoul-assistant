"""
Riichi Mahjong Shanten, Ukeire, Defense Danger Evaluation, and Nani-kiru Recommender.

Tile encoding:
- Base 34 index (0..33):
    0..8   : 1m..9m (万子)
    9..17  : 1p..9p (筒子)
    18..26 : 1s..9s (索子)
    27..33 : 1z..7z (東南西北白發中)
- String code (37 variants):
    '0m','0p','0s' = red fives (赤宝牌)
    '1m'..'9m', '1p'..'9p', '1s'..'9s', '1z'..'7z'
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple


TILE_CN = [
    '一万', '二万', '三万', '四万', '五万', '六万', '七万', '八万', '九万',
    '一筒', '二筒', '三筒', '四筒', '五筒', '六筒', '七筒', '八筒', '九筒',
    '一索', '二索', '三索', '四索', '五索', '六索', '七索', '八索', '九索',
    '东', '南', '西', '北', '白', '发', '中',
]

TERMINALS_AND_HONORS = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)


def code_to_34(code: str) -> int:
    num = int(code[0])
    suit = code[1]
    if num == 0:
        num = 5
    base = {'m': 0, 'p': 9, 's': 18, 'z': 27}[suit]
    return base + num - 1


def is_aka5(code: str) -> bool:
    return len(code) == 2 and code[0] == '0' and code[1] in ('m', 'p', 's')


def code_to_cn(code: str) -> str:
    if is_aka5(code):
        return {'0m': '赤五万', '0p': '赤五筒', '0s': '赤五索'}[code]
    return TILE_CN[code_to_34(code)]


def idx34_to_code(idx: int) -> str:
    if idx < 9:
        return f'{idx + 1}m'
    if idx < 18:
        return f'{idx - 8}p'
    if idx < 27:
        return f'{idx - 17}s'
    return f'{idx - 26}z'


def dora_indicator_to_dora34(ind_code: str, player_count: int = 4) -> int:
    idx = code_to_34(ind_code)
    if player_count == 3 and idx == 0:
        return 8  # 3-player mahjong: 1m indicator wraps directly to 9m
    if idx < 27:
        suit_base = (idx // 9) * 9
        return suit_base + ((idx % 9 + 1) % 9)
    if idx < 31:  # winds 27..30 -> E->S->W->N->E
        return 27 + ((idx - 27 + 1) % 4)
    # dragons 31..33 -> Wh->G->R->Wh
    return 31 + ((idx - 31 + 1) % 3)


@lru_cache(maxsize=65536)
def _suit_pareto(c: Tuple[int, ...]) -> Tuple[Tuple[int, int], ...]:
    """Return Pareto-optimal (mentsu, taatsu) pairs achievable from a 9-tile suit."""
    results: Dict[int, int] = {}

    def dfs(arr: List[int], i: int, m: int, t: int):
        while i < 9 and arr[i] == 0:
            i += 1
        if i >= 9:
            prev = results.get(m, -1)
            if t > prev:
                results[m] = t
            return

        # 1. Koutsu (triplet)
        if arr[i] >= 3:
            arr[i] -= 3
            dfs(arr, i, m + 1, t)
            arr[i] += 3

        # 2. Shuntsu (sequence)
        if i <= 6 and arr[i] >= 1 and arr[i + 1] >= 1 and arr[i + 2] >= 1:
            arr[i] -= 1
            arr[i + 1] -= 1
            arr[i + 2] -= 1
            dfs(arr, i, m + 1, t)
            arr[i] += 1
            arr[i + 1] += 1
            arr[i + 2] += 1

        # 3. Taatsu options (only need to try if not skipped, or try taking one + advancing)
        if arr[i] >= 2:
            arr[i] -= 2
            dfs(arr, i, m, t + 1)
            arr[i] += 2

        if i <= 7 and arr[i] >= 1 and arr[i + 1] >= 1:
            arr[i] -= 1
            arr[i + 1] -= 1
            dfs(arr, i, m, t + 1)
            arr[i] += 1
            arr[i + 1] += 1

        if i <= 6 and arr[i] >= 1 and arr[i + 2] >= 1:
            arr[i] -= 1
            arr[i + 2] -= 1
            dfs(arr, i, m, t + 1)
            arr[i] += 1
            arr[i + 2] += 1

        # 4. Skip isolated tile at i
        saved = arr[i]
        arr[i] = 0
        dfs(arr, i + 1, m, t)
        arr[i] = saved

    dfs(list(c), 0, 0, 0)
    # Prune dominated (m1 <= m2 and m1 + t1 <= m2 + t2)
    items = sorted(results.items(), reverse=True)
    pareto: List[Tuple[int, int]] = []
    best_sum = -1
    for m, t in items:
        if 2 * m + t > best_sum or (m + t > max((pm + pt for pm, pt in pareto), default=-1)):
            pareto.append((m, t))
            if 2 * m + t > best_sum:
                best_sum = 2 * m + t
    return tuple(pareto)


@lru_cache(maxsize=65536)
def _suit_all_mentsu(c: Tuple[int, ...]) -> bool:
    """Return True iff a 9-tile suit count tuple decomposes completely into mentsu."""
    if sum(c) % 3 != 0:
        return False
    arr = list(c)

    def dfs(i: int) -> bool:
        while i < 9 and arr[i] == 0:
            i += 1
        if i >= 9:
            return True
        if arr[i] >= 3:
            arr[i] -= 3
            if dfs(i):
                return True
            arr[i] += 3
        if i <= 6 and arr[i] >= 1 and arr[i + 1] >= 1 and arr[i + 2] >= 1:
            arr[i] -= 1
            arr[i + 1] -= 1
            arr[i + 2] -= 1
            if dfs(i):
                return True
            arr[i] += 1
            arr[i + 1] += 1
            arr[i + 2] += 1
        return False

    return dfs(0)


@lru_cache(maxsize=65536)
def _suit_mentsu_and_pair(c: Tuple[int, ...]) -> bool:
    """Return True iff a 9-tile suit count tuple decomposes into mentsu + 1 pair."""
    if sum(c) % 3 != 2:
        return False
    arr = list(c)
    for i in range(9):
        if arr[i] >= 2:
            arr[i] -= 2
            if _suit_all_mentsu(tuple(arr)):
                return True
            arr[i] += 2
    return False


def is_agari(counts34: List[int], num_melds: int = 0) -> bool:
    """Fast exact check for winning hand (shanten == -1)."""
    max_m = 4 - num_melds
    if max_m < 0:
        return False
    if sum(counts34) != 3 * max_m + 2:
        return calc_shanten(counts34, num_melds) < 0

    # 1. Chiitoi & Kokushi (closed hand only)
    if num_melds == 0:
        if sum(1 for c in counts34 if c == 2) == 7:
            return True
        if all(counts34[i] >= 1 for i in TERMINALS_AND_HONORS) and any(counts34[i] >= 2 for i in TERMINALS_AND_HONORS):
            return True

    # 2. Standard 4-mentsu + 1-pair
    honor_pairs = 0
    for i in range(27, 34):
        c = counts34[i]
        if c == 1 or c == 4:
            return False
        if c == 2:
            honor_pairs += 1
            if honor_pairs > 1:
                return False

    s0 = tuple(counts34[0:9])
    s1 = tuple(counts34[9:18])
    s2 = tuple(counts34[18:27])
    m0 = sum(s0) % 3
    m1 = sum(s1) % 3
    m2 = sum(s2) % 3
    if m0 == 1 or m1 == 1 or m2 == 1:
        return False
    if honor_pairs + (m0 == 2) + (m1 == 2) + (m2 == 2) != 1:
        return False

    if not (_suit_mentsu_and_pair(s0) if m0 == 2 else _suit_all_mentsu(s0)):
        return False
    if not (_suit_mentsu_and_pair(s1) if m1 == 2 else _suit_all_mentsu(s1)):
        return False
    if not (_suit_mentsu_and_pair(s2) if m2 == 2 else _suit_all_mentsu(s2)):
        return False
    return True


def shanten_standard(counts34: List[int], num_melds: int = 0) -> int:
    max_m = 4 - num_melds
    if max_m < 0:
        return 99

    h_m = 0
    h_t = 0
    for i in range(27, 34):
        c = counts34[i]
        if c >= 3:
            h_m += 1
            c -= 3
        if c >= 2:
            h_t += 1

    def best_no_pair(
        p0: Tuple[Tuple[int, int], ...],
        p1: Tuple[Tuple[int, int], ...],
        p2: Tuple[Tuple[int, int], ...],
        hm: int,
        ht: int,
    ) -> int:
        best = 99
        for m0, t0 in p0:
            for m1, t1 in p1:
                for m2, t2 in p2:
                    m = m0 + m1 + m2 + hm
                    if m > max_m:
                        m = max_m
                    t = min(t0 + t1 + t2 + ht, max_m - m)
                    sh = 2 * max_m - 2 * m - t
                    if sh < best:
                        best = sh
        return best

    p0_base = _suit_pareto(tuple(counts34[0:9]))
    p1_base = _suit_pareto(tuple(counts34[9:18]))
    p2_base = _suit_pareto(tuple(counts34[18:27]))
    best_sh = best_no_pair(p0_base, p1_base, p2_base, h_m, h_t)

    # Try each possible jantou (pair), recomputing only the modified suit's pareto tuple
    c_work = list(counts34)
    for i in range(34):
        if c_work[i] >= 2:
            c_work[i] -= 2
            if i < 9:
                sh = best_no_pair(_suit_pareto(tuple(c_work[0:9])), p1_base, p2_base, h_m, h_t) - 1
            elif i < 18:
                sh = best_no_pair(p0_base, _suit_pareto(tuple(c_work[9:18])), p2_base, h_m, h_t) - 1
            elif i < 27:
                sh = best_no_pair(p0_base, p1_base, _suit_pareto(tuple(c_work[18:27])), h_m, h_t) - 1
            else:
                c = c_work[i]
                hm = h_m - (1 if counts34[i] >= 3 and c < 3 else 0)
                old_t = 1 if (counts34[i] % 3) >= 2 else 0
                new_t = 1 if (c % 3) >= 2 else 0
                ht = h_t - old_t + new_t
                sh = best_no_pair(p0_base, p1_base, p2_base, hm, ht) - 1
            c_work[i] += 2
            if sh < best_sh:
                best_sh = sh
    return best_sh


def shanten_chiitoi(counts34: List[int]) -> int:
    pairs = sum(1 for c in counts34 if c >= 2)
    kinds = sum(1 for c in counts34 if c >= 1)
    return 6 - pairs + max(0, 7 - kinds)


def shanten_kokushi(counts34: List[int]) -> int:
    uniq = sum(1 for i in TERMINALS_AND_HONORS if counts34[i] >= 1)
    has_pair = any(counts34[i] >= 2 for i in TERMINALS_AND_HONORS)
    return 13 - uniq - (1 if has_pair else 0)


def calc_shanten(counts34: List[int], num_melds: int = 0) -> int:
    sh = shanten_standard(counts34, num_melds)
    if num_melds == 0:
        sh = min(sh, shanten_chiitoi(counts34), shanten_kokushi(counts34))
    return sh


def _can_complete_tenpai_tile(counts34: List[int], t: int, kokushi_tenpai: bool) -> bool:
    if counts34[t] > 0:
        return True
    if kokushi_tenpai and t in TERMINALS_AND_HONORS:
        return True
    if t < 27:
        n = t % 9
        if (
            (n >= 1 and counts34[t - 1] > 0)
            or (n >= 2 and counts34[t - 2] > 0)
            or (n <= 7 and counts34[t + 1] > 0)
            or (n <= 6 and counts34[t + 2] > 0)
        ):
            return True
    return False


def _count_tenpai_ukeire(
    counts34: List[int],
    num_melds: int,
    visible34: List[int],
    player_count: int = 4,
) -> int:
    """Fast total ukeire count for a tenpai (shanten == 0) hand using is_agari."""
    kokushi_tenpai = (num_melds == 0) and (shanten_kokushi(counts34) == 0)
    total = 0
    for t in range(34):
        if player_count == 3 and 1 <= t <= 7:
            continue
        if counts34[t] >= 4 or not _can_complete_tenpai_tile(counts34, t, kokushi_tenpai):
            continue
        counts34[t] += 1
        won = is_agari(counts34, num_melds)
        counts34[t] -= 1
        if won:
            total += max(0, 4 - visible34[t])
    return total


def calc_ukeire(
    counts34: List[int],
    num_melds: int,
    visible34: List[int],
    cur_shanten: Optional[int] = None,
    player_count: int = 4,
) -> Tuple[int, int, List[Tuple[int, int]]]:
    """
    Return (shanten, total_live_ukeire, [(tile34, live_rem), ...]) for a 13-3*m tile hand.
    """
    sh = calc_shanten(counts34, num_melds) if cur_shanten is None else cur_shanten
    if sh < 0:
        return sh, 0, []
    total = 0
    tiles: List[Tuple[int, int]] = []
    if sh == 0:
        kokushi_tenpai = (num_melds == 0) and (shanten_kokushi(counts34) == 0)
        for t in range(34):
            if player_count == 3 and 1 <= t <= 7:
                continue  # 2m..8m do not exist in 3-player mahjong
            if counts34[t] >= 4 or not _can_complete_tenpai_tile(counts34, t, kokushi_tenpai):
                continue
            counts34[t] += 1
            won = is_agari(counts34, num_melds)
            counts34[t] -= 1
            if won:
                rem = max(0, 4 - visible34[t])
                total += rem
                if rem > 0:
                    tiles.append((t, rem))
        return sh, total, tiles

    for t in range(34):
        if player_count == 3 and 1 <= t <= 7:
            continue  # 2m..8m do not exist in 3-player mahjong
        if counts34[t] >= 4:
            continue
        counts34[t] += 1
        ns = calc_shanten(counts34, num_melds)
        counts34[t] -= 1
        if ns < sh:
            rem = max(0, 4 - visible34[t])
            total += rem
            if rem > 0:
                tiles.append((t, rem))
    return sh, total, tiles


def format_shanten(sh: int) -> str:
    if sh < 0:
        return '和牌 (-1向听)'
    if sh == 0:
        return '听牌 (0向听)'
    return f'{sh}向听'


@dataclass
class BoardState:
    hand: List[str]                          # 1..14 tiles in self hand
    dora_indicators: List[str] = field(default_factory=list)
    rivers: List[List[str]] = field(default_factory=lambda: [[], [], [], []])   # 0=self,1=right,2=top,3=left
    melds: List[List[List[str]]] = field(default_factory=lambda: [[], [], [], []])
    riichi: List[bool] = field(default_factory=lambda: [False, False, False, False])
    theme: str = 'auto'
    player_count: int = 4                    # 3 (三麻) or 4 (四麻)
    kita: List[int] = field(default_factory=lambda: [0, 0, 0, 0])  # peeled North (拔北) counts per player


@dataclass
class CandidateEval:
    code: str
    idx34: int
    shanten: int
    ukeire: int
    ukeire_tiles: List[Tuple[int, int]]
    two_step_score: float
    furiten: bool
    danger: float
    danger_tags: List[str]
    eff_score: float
    balanced_score: float
    est_han: float = 1.0
    est_points: int = 1000
    points_score: float = 0.0
    eff_points_score: float = 0.0
    def_points_score: float = 0.0
    yaku_tags: List[str] = field(default_factory=list)


@dataclass
class RecommendationResult:
    current_shanten: int
    efficiency_best: Optional[CandidateEval]
    defense_best: Optional[CandidateEval]
    balanced_best: Optional[CandidateEval]
    points_best: Optional[CandidateEval]
    eff_points_best: Optional[CandidateEval]
    def_points_best: Optional[CandidateEval]
    candidates: List[CandidateEval]
    dora_tiles34: List[int]
    visible34: List[int]
    can_riichi: bool = False
    riichi_eff_best: Optional[CandidateEval] = None
    riichi_def_best: Optional[CandidateEval] = None
    riichi_points_best: Optional[CandidateEval] = None
    summary_note: str = ''


def build_visible34(board: BoardState) -> List[int]:
    vis = [0] * 34
    for c in board.hand:
        vis[code_to_34(c)] += 1
    for c in board.dora_indicators:
        vis[code_to_34(c)] += 1
    for rv in board.rivers:
        for c in rv:
            vis[code_to_34(c)] += 1
    for pm in board.melds:
        for grp in pm:
            for c in grp:
                vis[code_to_34(c)] += 1
    for k_cnt in board.kita:
        vis[30] += max(0, int(k_cnt))  # 30 = 4z (北)
    return [min(4, x) for x in vis]


def _single_opponent_danger(
    tile34: int,
    is_aka: bool,
    opp_river34: set,
    opp_riichi: bool,
    opp_river_len: int,
    opp_melds: int,
    visible34: List[int],
    dora_set34: set,
    opp_kita: int = 0,
    player_count: int = 4,
) -> Tuple[float, List[str]]:
    """
    Evaluate danger [0..100] of discarding `tile34` against one opponent.
    Uses: Genbutsu, 0-copy/Kabe (No-Chance & One-Chance), Suji, Honor visibility, 3P terminal rules, and Dora proximity.
    """
    tags: List[str] = []
    if tile34 in opp_river34:
        return 0.0, ['现物']

    # Copies potentially held by others (4 - total visible on board & our hand)
    rem_out = max(0, 4 - visible34[tile34])

    # Honor tile (27..33) OR 1m/9m in 3-player mahjong (cannot form sequences)
    is_isolated_3p_man = (player_count == 3 and tile34 in (0, 8))
    if tile34 >= 27 or is_isolated_3p_man:
        kind_lbl = '幺九' if is_isolated_3p_man else '字牌'
        if rem_out == 0:
            return 0.0, ['绝张(4见)']
        if rem_out == 1:
            base = 5.0
            tags.append(f'3枚见{kind_lbl}')
        elif rem_out == 2:
            base = 16.0
            tags.append(f'2枚见{kind_lbl}')
        elif rem_out == 3:
            base = 28.0
            tags.append(f'1枚见{kind_lbl}')
        else:
            base = 44.0
            tags.append(f'生牌{kind_lbl}')
        if tile34 in dora_set34:
            base += 12.0
            tags.append('宝牌')
        if opp_kita > 0:
            tags.append(f'拔北×{opp_kita}')
        return min(100.0, base), tags

    # Number tile (0..26)
    suit_base = (tile34 // 9) * 9
    n = (tile34 % 9) + 1  # 1..9

    # Check Suji against opponent's river
    left_suji = (n >= 4) and ((suit_base + (n - 4)) in opp_river34)
    right_suji = (n <= 6) and ((suit_base + (n + 2)) in opp_river34)

    if n in (1, 9):
        suji_safe = left_suji if n == 9 else right_suji
        if suji_safe:
            base = 13.0
            tags.append('外筋19')
        else:
            base = 38.0
            tags.append('无筋19')
    elif n in (2, 8):
        suji_safe = left_suji if n == 8 else right_suji
        if suji_safe:
            base = 18.0
            tags.append('筋28')
        else:
            base = 48.0
            tags.append('无筋28')
    elif n in (3, 7):
        suji_safe = left_suji if n == 7 else right_suji
        if suji_safe:
            base = 22.0
            tags.append('筋37')
        else:
            base = 58.0
            tags.append('无筋37')
    else:  # 4, 5, 6
        if left_suji and right_suji:
            base = 18.0
            tags.append('両筋456')
        elif left_suji or right_suji:
            base = 46.0
            tags.append('片筋456')
        else:
            base = 68.0
            tags.append('无筋456')

    # Check Kabe (Wall / No-Chance / One-Chance) on sequences containing n
    rem_suit = [max(0, 4 - visible34[suit_base + k]) for k in range(9)]
    seq_min_rem = []
    idx0 = n - 1
    if idx0 >= 2:
        seq_min_rem.append(min(rem_suit[idx0 - 2], rem_suit[idx0 - 1]))  # (n-2, n-1, n)
    if 1 <= idx0 <= 7:
        seq_min_rem.append(min(rem_suit[idx0 - 1], rem_suit[idx0 + 1]))  # (n-1, n, n+1)
    if idx0 <= 6:
        seq_min_rem.append(min(rem_suit[idx0 + 1], rem_suit[idx0 + 2]))  # (n, n+1, n+2)

    if all(r == 0 for r in seq_min_rem):
        # Complete Wall (No Chance) - can only lose to tanki / shanpon!
        if rem_out == 0:
            return 0.0, ['绝张(4见)']
        base = min(base, 7.0 if rem_out == 1 else 11.0)
        tags.append('完全壁')
    elif all(r <= 1 for r in seq_min_rem):
        base *= 0.65
        tags.append('半壁(1-Chance)')

    if rem_out == 0:
        # Cannot lose to tanki/shanpon/ryanmen? Wait: ryanmen needs only neighbors, NOT tile34 itself!
        # Wait: if rem_out == 0 (all 4 in our hand/visible), opponent CAN STILL RON on ryanmen/kanchan/penchan!
        # Only tanki/shanpon are impossible. So do NOT return 0 for number tiles when rem_out == 0 unless complete wall!
        base *= 0.90

    # Dora / Red-5 / Dora-adjacent penalty
    if tile34 in dora_set34:
        base += 14.0
        tags.append('宝牌')
    elif is_aka:
        base += 6.0
        tags.append('赤宝牌')
    elif any(abs(tile34 - d) == 1 and (tile34 // 9 == d // 9) and d < 27 for d in dora_set34):
        base += 5.0
        tags.append('宝牌旁张')

    if opp_kita > 0:
        tags.append(f'拔北×{opp_kita}')

    return min(100.0, base), tags


def evaluate_danger(tile34: int, is_aka: bool, board: BoardState, visible34: List[int], dora_set34: set) -> Tuple[float, List[str]]:
    opp_scores = []
    all_tags: List[str] = []
    weights = []

    active_opps = (1, 2) if board.player_count == 3 else (1, 2, 3)
    for p in active_opps:
        rv_set = {code_to_34(c) for c in board.rivers[p]}
        rv_len = len(board.rivers[p])
        m_cnt = len(board.melds[p])
        is_r = board.riichi[p]
        opp_kita = board.kita[p] if p < len(board.kita) else 0
        d_p, tags_p = _single_opponent_danger(
            tile34, is_aka, rv_set, is_r, rv_len, m_cnt, visible34, dora_set34,
            opp_kita=opp_kita, player_count=board.player_count,
        )
        # Threat weight: Riichi = 1.0, active late/melded/kita player scales up
        if is_r:
            w = 1.0
        else:
            w = min(0.90, 0.18 + 0.025 * rv_len + 0.12 * m_cnt + 0.07 * opp_kita)
        opp_scores.append((d_p, w, p, tags_p, is_r))
        weights.append(w)

    if not opp_scores:
        return 0.0, []

    # Combined threat: emphasize max weighted danger + weighted average across active opponents
    max_wd = max(d * w for d, w, _, _, _ in opp_scores)
    avg_wd = sum(d * w for d, w, _, _, _ in opp_scores) / max(1e-6, sum(weights))
    combined = 0.65 * max_wd + 0.35 * avg_wd

    # Pick informative tags from the highest-threat opponent (or '全员安牌' if all 0)
    if all(d == 0.0 for d, _, _, _, _ in opp_scores):
        all_tags = ['全员安牌']
    else:
        top = max(opp_scores, key=lambda x: (x[4], x[1] * x[0]))
        all_tags = list(top[3])
        if top[4]:
            all_tags.insert(0, '对立直')
    return round(combined, 2), all_tags


def _estimate_hand_value(
    c13: List[int],
    board: BoardState,
    discard_code: str,
    dora_set34: set,
    sh: int,
    num_melds: int,
    player_count: int,
) -> Tuple[float, int, List[str]]:
    """Estimate expected han, points, and yaku tags for the remaining 13-tile hand."""
    tags: List[str] = []
    han = 0.0

    # 1. Dora & Akadora remaining in hand (plus melds)
    aka_kept = sum(1 for c in board.hand if is_aka5(c)) - (1 if is_aka5(discard_code) else 0)
    dora_kept = sum(c13[d] for d in dora_set34)
    for grp in board.melds[0]:
        for c in grp:
            if is_aka5(c):
                aka_kept += 1
            if code_to_34(c) in dora_set34:
                dora_kept += 1
    kita_kept = board.kita[0] if player_count == 3 else 0
    total_dora = max(0, aka_kept) + max(0, dora_kept) + max(0, kita_kept)
    if total_dora > 0:
        han += float(total_dora)
        tags.append(f'宝牌×{total_dora}')

    # 2. Menzen / Riichi potential
    if num_melds == 0:
        han += 1.2  # Riichi + tsumo/ura potential
        tags.append('门清立直')

    # 3. Tanyao (断幺九)
    term_honor_cnt = sum(c13[i] for i in (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33))
    meld_has_term = any(
        code_to_34(c) in (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)
        for grp in board.melds[0] for c in grp
    )
    if not meld_has_term:
        if term_honor_cnt == 0:
            han += 1.0
            tags.append('断幺九')
        elif term_honor_cnt <= 2:
            han += 0.35

    # 4. Yakuhai (三元牌 & 风牌)
    yakuhai_han = 0.0
    for z in (31, 32, 33):  # 白 发 中
        if c13[z] >= 3:
            yakuhai_han += 1.0
        elif c13[z] == 2:
            yakuhai_han += 0.45
    for z in (27, 28):  # 东 南
        if c13[z] >= 3:
            yakuhai_han += 0.8
        elif c13[z] == 2:
            yakuhai_han += 0.25
    if yakuhai_han > 0:
        han += yakuhai_han
        tags.append('役牌')

    # 5. Flush (清一色 / 混一色)
    suit_cnts = [sum(c13[s * 9:(s + 1) * 9]) for s in range(3)]
    honor_cnt = sum(c13[27:34])
    meld_suits = set()
    meld_has_honor = False
    for grp in board.melds[0]:
        for c in grp:
            idx = code_to_34(c)
            if idx >= 27:
                meld_has_honor = True
            else:
                meld_suits.add(idx // 9)
    max_suit_idx = max(range(3), key=lambda s: suit_cnts[s])
    other_suit_cnt = sum(suit_cnts[s] for s in range(3) if s != max_suit_idx)
    if len(meld_suits) <= 1 and (not meld_suits or max_suit_idx in meld_suits):
        if other_suit_cnt == 0:
            if honor_cnt == 0 and not meld_has_honor:
                han += 6.0 if num_melds == 0 else 5.0
                tags.append('清一色')
            else:
                han += 3.0 if num_melds == 0 else 2.0
                tags.append('混一色')
        elif other_suit_cnt <= 2:
            han += 0.8
            tags.append('一色倾向')

    # 6. Chiitoitsu (七对子)
    if num_melds == 0 and shanten_chiitoi(c13) <= max(1, sh):
        pairs = sum(1 for x in c13 if x >= 2)
        if pairs >= 4:
            han = max(han, 2.0 + total_dora)
            tags.append('七对子')

    han = max(1.0, han)
    # Approximate Ron/Tsumo points table (non-dealer)
    if han < 1.5:
        pts = 1000
    elif han < 2.5:
        pts = 2000
    elif han < 3.5:
        pts = 3900
    elif han < 4.5:
        pts = 7700
    elif han < 6.0:
        pts = 8000
    elif han < 8.0:
        pts = 12000
    elif han < 11.0:
        pts = 16000
    else:
        pts = 24000
    return round(han, 1), pts, tags


def recommend_discards(board: BoardState) -> RecommendationResult:
    p_cnt = 3 if board.player_count == 3 else 4
    num_melds = len(board.melds[0])
    dora_tiles34 = [dora_indicator_to_dora34(ind, player_count=p_cnt) for ind in board.dora_indicators]
    dora_set34 = set(dora_tiles34)
    visible34 = build_visible34(board)

    hand_counts34 = [0] * 34
    for c in board.hand:
        hand_counts34[code_to_34(c)] += 1

    # Handle both 14-tile (3k+2, discard turn) and 13-tile (3k+1, waiting) gracefully
    cur_sh = calc_shanten(hand_counts34, num_melds)
    self_river34 = {code_to_34(c) for c in board.rivers[0]}
    kita_note = '手牌含北，三麻可优先拔北' if (p_cnt == 3 and '4z' in board.hand) else ''

    # Distinct candidate codes in hand (keep 0m vs 5m distinct!)
    seen_codes = set()
    candidate_codes: List[str] = []
    for c in board.hand:
        if c not in seen_codes:
            seen_codes.add(c)
            candidate_codes.append(c)

    if len(board.hand) % 3 != 2:
        # Not discard turn (e.g. 13 tiles): still compute current shanten & ukeire for display
        sh, uk, uk_tiles = calc_ukeire(list(hand_counts34), num_melds, visible34, cur_sh, player_count=p_cnt)
        note = f'当前手牌共 {len(board.hand)} 张（非摸打切牌回合），当前向听：{format_shanten(sh)}，有效进张 {uk} 枚'
        if kita_note:
            note = f'{note}；{kita_note}'
        return RecommendationResult(
            current_shanten=sh,
            efficiency_best=None,
            defense_best=None,
            balanced_best=None,
            points_best=None,
            eff_points_best=None,
            def_points_best=None,
            candidates=[],
            dora_tiles34=dora_tiles34,
            visible34=visible34,
            can_riichi=False,
            summary_note=note,
        )

    raw_evals: List[CandidateEval] = []
    total_dora_in_hand = sum(1 for c in board.hand if is_aka5(c) or code_to_34(c) in dora_set34)

    for code in candidate_codes:
        idx = code_to_34(code)
        aka = is_aka5(code)
        c13 = list(hand_counts34)
        c13[idx] -= 1

        sh, uk, uk_tiles = calc_ukeire(c13, num_melds, visible34, player_count=p_cnt)

        # Furiten check if tenpai
        furiten = (sh == 0) and any(t in self_river34 for t, _ in uk_tiles)

        # 2-step / good-shape tiebreaker
        two_step = 0.0
        if sh == 0:
            # Prefer non-furiten and wider wait
            two_step = float(uk) * (0.25 if furiten else 1.0)
        elif sh == 1 and uk > 0:
            # Expected tenpai ukeire on next draw
            weighted_next = 0.0
            for t, rem in uk_tiles:
                c13[t] += 1
                # Best discard after drawing t (check if discarding d2 leaves tenpai before counting waits)
                best_next_uk = 0
                for d2 in range(34):
                    if c13[d2] > 0:
                        c13[d2] -= 1
                        if calc_shanten(c13, num_melds) == 0:
                            uk2 = _count_tenpai_ukeire(c13, num_melds, visible34, player_count=p_cnt)
                            if uk2 > best_next_uk:
                                best_next_uk = uk2
                        c13[d2] += 1
                c13[t] -= 1
                weighted_next += rem * best_next_uk
            two_step = weighted_next / max(1, uk)
        else:
            # Slight preference for keeping middle tiles / connected shapes
            two_step = float(uk)

        # Dora & Aka retention bonus (higher = better to KEEP, i.e., penalty to discard)
        is_kita_north = (p_cnt == 3 and code == '4z')
        dora_lost = (1.5 if aka else 0.0) + (2.0 if idx in dora_set34 else 0.0) + (2.0 if is_kita_north else 0.0)
        _ = total_dora_in_hand

        danger, d_tags = evaluate_danger(idx, aka, board, visible34, dora_set34)
        if is_kita_north:
            d_tags = list(d_tags) + ['可拔北(+1宝牌)']

        # Pure efficiency score (higher is better)
        furiten_pen = -25.0 if furiten else 0.0
        eff_score = -sh * 1000.0 + uk * 10.0 + two_step * 0.45 - dora_lost * 2.5 + furiten_pen

        est_han, est_points, yaku_tags = _estimate_hand_value(
            c13, board, code, dora_set34, sh, num_melds, p_cnt
        )
        points_score = est_points - sh * 1400.0 + uk * 35.0 - (1500.0 if furiten else 0.0)
        eff_points_score = eff_score + est_points * 0.08
        def_points_score = est_points * 0.06 - danger * 18.0 - sh * 220.0

        raw_evals.append(CandidateEval(
            code=code,
            idx34=idx,
            shanten=sh,
            ukeire=uk,
            ukeire_tiles=uk_tiles,
            two_step_score=round(two_step, 2),
            furiten=furiten,
            danger=danger,
            danger_tags=d_tags,
            eff_score=round(eff_score, 2),
            balanced_score=0.0,
            est_han=est_han,
            est_points=est_points,
            points_score=round(points_score, 1),
            eff_points_score=round(eff_points_score, 1),
            def_points_score=round(def_points_score, 1),
            yaku_tags=yaku_tags,
        ))

    if not raw_evals:
        return RecommendationResult(
            cur_sh, None, None, None, None, None, None, [], dora_tiles34, visible34,
            can_riichi=False, summary_note=kita_note
        )

    best_sh = min(e.shanten for e in raw_evals)
    best_uk_at_min_sh = max(e.ukeire for e in raw_evals if e.shanten == best_sh)
    active_opps = (1, 2) if p_cnt == 3 else (1, 2, 3)
    any_riichi = any(board.riichi[p] for p in active_opps)
    max_turn = max((len(board.rivers[p]) for p in active_opps), default=0)

    # Adaptive defense weight for Balanced (牌效+防御)
    if any_riichi:
        def_w = 0.55 if best_sh == 0 else (1.05 if best_sh == 1 else 1.45)
    else:
        def_w = 0.25 + min(0.45, max_turn * 0.025)

    for e in raw_evals:
        sh_loss = (e.shanten - best_sh) * 32.0
        uk_loss = max(0, best_uk_at_min_sh - e.ukeire) * 1.4 if e.shanten == best_sh else 0.0
        is_kita_north = (p_cnt == 3 and e.code == '4z')
        dora_pen = (3.5 if is_aka5(e.code) else 0.0) + (5.0 if e.idx34 in dora_set34 else 0.0) + (5.0 if is_kita_north else 0.0)
        fur_pen = 18.0 if e.furiten else 0.0
        shape_bonus = min(8.0, e.two_step_score * 0.25)
        e.balanced_score = round(100.0 - sh_loss - uk_loss - dora_pen - fur_pen + shape_bonus - def_w * e.danger, 2)

    eff_best = max(raw_evals, key=lambda e: (e.eff_score, -e.danger))
    def_best = min(raw_evals, key=lambda e: (e.danger, e.shanten, -e.ukeire, -e.eff_score))
    bal_best = max(raw_evals, key=lambda e: (e.balanced_score, e.eff_score))
    pts_best = max(raw_evals, key=lambda e: (e.points_score, e.est_points, e.eff_score))
    eff_pts_best = max(raw_evals, key=lambda e: (e.eff_points_score, e.eff_score))
    def_pts_best = max(raw_evals, key=lambda e: (e.def_points_score, -e.danger, e.est_points))

    # Check if self has a Riichi opportunity (menzen, not already riichi, at least one tenpai discard)
    tenpai_cands = [e for e in raw_evals if e.shanten == 0]
    can_riichi = (num_melds == 0) and (not board.riichi[0]) and len(tenpai_cands) > 0
    riichi_eff_best = None
    riichi_def_best = None
    riichi_pts_best = None
    if can_riichi:
        riichi_eff_best = max(tenpai_cands, key=lambda e: (e.ukeire, -int(e.furiten), e.eff_score, -e.danger))
        riichi_def_best = min(tenpai_cands, key=lambda e: (e.danger, int(e.furiten), -e.ukeire))
        riichi_pts_best = max(tenpai_cands, key=lambda e: (e.est_points, e.ukeire, -int(e.furiten)))

    candidates_sorted = sorted(raw_evals, key=lambda e: (-e.balanced_score, -e.eff_score))

    return RecommendationResult(
        current_shanten=cur_sh,
        efficiency_best=eff_best,
        defense_best=def_best,
        balanced_best=bal_best,
        points_best=pts_best,
        eff_points_best=eff_pts_best,
        def_points_best=def_pts_best,
        candidates=candidates_sorted,
        dora_tiles34=dora_tiles34,
        visible34=visible34,
        can_riichi=can_riichi,
        riichi_eff_best=riichi_eff_best,
        riichi_def_best=riichi_def_best,
        riichi_points_best=riichi_pts_best,
        summary_note=kita_note,
    )
