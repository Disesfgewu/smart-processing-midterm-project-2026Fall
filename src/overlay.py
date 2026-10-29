"""把偵測結果畫到 BGR 影像上（遮罩 / 輪廓 / 框 / 各提示詞數量與面積 / FPS）。app.py 與伺服器攝影機共用。"""
import cv2
import numpy as np

COLORS = [(0, 200, 0), (0, 128, 255), (255, 80, 80), (200, 0, 200), (0, 200, 200), (200, 200, 0)]  # BGR


def draw_overlay(frame, dets, prompts, timing=None, fps=None, style="both", expected=None):
    """style: mask | contour | both | box。expected: {提示詞: 預期數量}，有設定時顯示 OK/NG。"""
    color = lambda label: COLORS[prompts.index(label) % len(COLORS)]
    if style in ("mask", "both"):
        overlay = frame.copy()
        for d in dets:
            if d.mask is not None:
                overlay[d.mask] = color(d.label)
        a = 0.35 if style == "both" else 0.45
        frame = cv2.addWeighted(overlay, a, frame, 1 - a, 0)
    for d in dets:
        c = color(d.label)
        x0, y0, x1, y1 = [int(v) for v in d.box]
        if d.mask is None:  # 沒有遮罩（只要框模式）一律畫框
            cv2.rectangle(frame, (x0, y0), (x1, y1), c, 2)
        elif style in ("contour", "both"):
            cv2.polylines(frame, [pts.reshape(-1, 1, 2).astype(np.int32) for pts in d.contours()], True, c, 2, cv2.LINE_AA)
        if d.mask is not None and style in ("box", "both"):
            cv2.rectangle(frame, (x0, y0), (x1, y1), c, 1 if style == "both" else 2)
        cv2.putText(frame, f"{d.label} {d.score:.2f}", (x0, max(y0 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    y = 20
    for p in prompts:
        mine = [d for d in dets if d.label == p]
        text = f"{p}: count={len(mine)}" + (f" area={int(sum(d.area for d in mine))}px" if all(d.mask is not None for d in mine) else "")
        if expected and p in expected:
            ok = len(mine) == expected[p]
            text += f"  [{'OK' if ok else 'NG'} expect {expected[p]}]"
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color(p), 2, cv2.LINE_AA)
        y += 22
    if fps is not None:
        t = timing or {}
        cv2.putText(frame, f"{fps:.1f} FPS  owl {t.get('owl_ms', 0):.0f}ms  sam {t.get('sam_ms', 0):.0f}ms",
                    (8, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return frame
