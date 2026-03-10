"""
引出線（リーダーライン）検出

青マスク上のBFS（幅優先探索）で引出線の実際の経路を辿り、
paragraph と スリーブ円のペアリングを行う。
L字型・折れ曲がった引出線にも対応。
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from models import BBox, OcrParagraph, PixelPoint


@dataclass
class SleeveTextLink:
    """引出線によるスリーブ円⇔paragraphの紐付け"""
    sleeve_idx: int
    paragraph_idx: int


# スリーブ注釈っぽいparagraphのフィルタ
_SLEEVE_HINT = re.compile(
    r"(SK|sl|排水|給水|消火|通気|雑排|汚水|雨水|排|給|\d+A|\d+[Φφ])",
    re.IGNORECASE,
)


def _make_blue_mask(
    img_bgr: np.ndarray,
    paragraphs: list[OcrParagraph] | None = None,
) -> np.ndarray:
    """
    青色マスクを生成。
    paragraphsが指定された場合、テキスト領域を消去して
    引出線だけが残るようにする。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([135, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # テキスト領域の内側だけ消去（外周フリンジは残して経路接続を保つ）
    # これにより青テキスト経由のショートカットを防ぎつつ、
    # 引出線→テキストの接続は維持
    fringe = 8  # 外周に残すピクセル数
    if paragraphs:
        for para in paragraphs:
            bx = max(0, int(para.bbox.x) + fringe)
            by = max(0, int(para.bbox.y) + fringe)
            bx2 = min(mask.shape[1], int(para.bbox.x + para.bbox.w) - fringe)
            by2 = min(mask.shape[0], int(para.bbox.y + para.bbox.h) - fringe)
            if bx < bx2 and by < by2:
                mask[by:by2, bx:bx2] = 0

    # 線の途切れを補完
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _bfs_from_point(
    mask: np.ndarray,
    start_x: int,
    start_y: int,
    max_dist: int = 800,
) -> np.ndarray:
    """
    青マスク上でBFSを実行し、各ピクセルへの経路距離マップを返す。
    到達できないピクセルは-1。
    """
    h, w = mask.shape
    dist_map = np.full((h, w), -1, dtype=np.int32)

    # 開始点が青マスク上にない場合、周囲を探す
    sy, sx = start_y, start_x
    if not (0 <= sy < h and 0 <= sx < w and mask[sy, sx] > 0):
        found = False
        for r in range(1, 20):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] > 0:
                        sy, sx = ny, nx
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            return dist_map

    dist_map[sy, sx] = 0
    queue = deque([(sy, sx)])

    # 8方向近傍
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while queue:
        cy, cx = queue.popleft()
        cd = dist_map[cy, cx]
        if cd >= max_dist:
            continue

        for dy, dx in dirs:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] > 0 and dist_map[ny, nx] == -1:
                dist_map[ny, nx] = cd + 1
                queue.append((ny, nx))

    return dist_map


def _min_bfs_dist_to_bbox(
    dist_map: np.ndarray,
    bbox: BBox,
    margin: float = 15.0,
) -> int:
    """BFS距離マップ上でBBox周辺の最小距離を返す。到達不可なら-1。
    テキスト領域は青マスクから除去されているので、
    BBox周辺のマージンで引出線の端を拾う。
    """
    h, w = dist_map.shape
    x1 = max(0, int(bbox.x - margin))
    y1 = max(0, int(bbox.y - margin))
    x2 = min(w, int(bbox.x + bbox.w + margin))
    y2 = min(h, int(bbox.y + bbox.h + margin))

    region = dist_map[y1:y2, x1:x2]
    reachable = region[region >= 0]
    if len(reachable) == 0:
        return -1
    return int(reachable.min())


def _hungarian(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    簡易ハンガリアン法（Munkres）。
    cost: n x n のコスト行列。
    Returns: (row_indices, col_indices)
    """
    n = cost.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = -1

            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]

    row_ind = []
    col_ind = []
    for j in range(1, n + 1):
        if p[j] != 0:
            row_ind.append(p[j] - 1)
            col_ind.append(j - 1)

    return np.array(row_ind), np.array(col_ind)


def detect_leader_lines_and_link(
    img_bgr: np.ndarray,
    sleeve_detections: list,
    paragraphs: list[OcrParagraph],
) -> list[SleeveTextLink]:
    """
    青マスク上のBFSでスリーブ円とparagraphをペアリング。

    各スリーブ中心からBFSで青ピクセルを辿り、
    到達可能なparagraphのうち経路距離が最短のものとリンク。
    L字型の引出線も正しく辿れる。
    """
    # 小さすぎるスリーブ検出を除外（テキスト上の誤検出）
    min_radius = 7.0
    valid_indices = [i for i, d in enumerate(sleeve_detections) if d.circle.radius_px >= min_radius]

    # テキスト領域を除去したマスクでBFS（引出線のみ辿る）
    mask = _make_blue_mask(img_bgr, paragraphs=paragraphs)

    # スリーブ注釈っぽいparagraphだけを候補に
    candidate_paras: list[int] = []
    for pi, para in enumerate(paragraphs):
        if len(para.content.strip()) < 4:
            continue
        if _SLEEVE_HINT.search(para.content):
            candidate_paras.append(pi)

    # 各スリーブからBFSして、各paragraphへの経路距離を計算
    @dataclass
    class _Candidate:
        sleeve_idx: int
        paragraph_idx: int
        bfs_dist: int

    candidates: list[_Candidate] = []

    for si in valid_indices:
        det = sleeve_detections[si]
        cx = int(round(det.circle.center_px.x))
        cy = int(round(det.circle.center_px.y))

        dist_map = _bfs_from_point(mask, cx, cy)

        for pi in candidate_paras:
            para = paragraphs[pi]
            bfs_d = _min_bfs_dist_to_bbox(dist_map, para.bbox)
            if bfs_d < 0:
                continue
            candidates.append(_Candidate(
                sleeve_idx=si,
                paragraph_idx=pi,
                bfs_dist=bfs_d,
            ))

    # ハンガリアン法で全体最適割り当て
    # BFS距離コスト行列を構築
    sleeve_ids = sorted(set(c.sleeve_idx for c in candidates))
    para_ids = sorted(set(c.paragraph_idx for c in candidates))

    if not sleeve_ids or not para_ids:
        return []

    s_idx_map = {s: i for i, s in enumerate(sleeve_ids)}
    p_idx_map = {p: i for i, p in enumerate(para_ids)}

    INF = 10_000_000
    n = max(len(sleeve_ids), len(para_ids))
    cost = np.full((n, n), INF, dtype=np.float64)

    for c in candidates:
        si = s_idx_map[c.sleeve_idx]
        pi = p_idx_map[c.paragraph_idx]
        # 同じペアで複数候補がある場合は最小距離を採用
        if c.bfs_dist < cost[si, pi]:
            cost[si, pi] = c.bfs_dist

    row_ind, col_ind = _hungarian(cost)

    links: list[SleeveTextLink] = []
    for r, c_idx in zip(row_ind, col_ind):
        if cost[r, c_idx] >= INF:
            continue
        if r < len(sleeve_ids) and c_idx < len(para_ids):
            links.append(SleeveTextLink(
                sleeve_idx=sleeve_ids[r],
                paragraph_idx=para_ids[c_idx],
            ))

    return links
