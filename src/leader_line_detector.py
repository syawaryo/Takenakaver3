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


def _make_blue_mask(img_bgr: np.ndarray) -> np.ndarray:
    """青色マスクを生成"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([135, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

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
    margin: float = 5.0,
) -> int:
    """BFS距離マップ上でBBox内の最小距離を返す。到達不可なら-1。"""
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
    mask = _make_blue_mask(img_bgr)

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

    for si, det in enumerate(sleeve_detections):
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

    # BFS距離が短い順にgreedy割り当て
    candidates.sort(key=lambda c: c.bfs_dist)

    links: list[SleeveTextLink] = []
    linked_sleeves: set[int] = set()
    linked_paragraphs: set[int] = set()

    for c in candidates:
        if c.sleeve_idx in linked_sleeves or c.paragraph_idx in linked_paragraphs:
            continue
        links.append(SleeveTextLink(sleeve_idx=c.sleeve_idx, paragraph_idx=c.paragraph_idx))
        linked_sleeves.add(c.sleeve_idx)
        linked_paragraphs.add(c.paragraph_idx)

    return links
