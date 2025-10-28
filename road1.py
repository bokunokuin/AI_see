# lane_paint_lanewidth_fixed.py
# 逐幀偵測左右車道直線 → 用四邊形鋪綠（含白線） → 輸出影片（含 debug 與主線黑白）

import cv2
import numpy as np
from collections import deque

# ===== [設定] 輸入/輸出與顯示 =====
VIDEO_IN  = "LaneVideo.mp4"          # 你的影片檔名
VIDEO_OUT = "lane_out.mp4"           # 鋪綠輸出檔
SHOW_PREVIEW = True                  # 是否顯示即時視窗
DEBUG_OUT = "lines_debug.mp4"        # 黑白清理後的影片
SAVE_DEBUG = True
CONNECTED_OUT = "lines_connected.mp4"  # 只畫左右主線的黑白影片
SAVE_CONNECTED = True
LINE_DRAW_THICKNESS = 6              # 主線（黑白輸出）粗細
WHITE_LINE_THICKNESS = 4             # 車道白線粗細
GREEN_TOP_RATIO = 0.82               # 只控制綠色顯示高度（不影響偵測）

# ===== [設定] 防爆條件 =====
MAX_BOTTOM_GROWTH = 1.5
REQUIRE_TOP_SMALLER = True
ENFORCE_TOP_X2_LT_BOTTOM = True
TOP_X_MULTIPLIER = 2.0

# ===== [設定] ROI 與偵測參數 =====
ROI_TOP_RATIO = 0.75                 # 只用畫面 y >= h*0.75 區域
BAND_BOTTOM_RATIO = 0.85             # 線段須觸碰底部帶

CANNY_LOW, CANNY_HIGH = 60, 150
HOUGH_RHO, HOUGH_THETA = 1, np.pi/180
HOUGH_THRESHOLD = 40
HOUGH_MIN_LINE_LENGTH = 30
HOUGH_MAX_LINE_GAP    = 60

# 形態學清雜訊
K_OPEN  = (1, 2)    # 開運算 kernel（去小白噪）
K_CLOSE = (5, 5)    # 閉運算 kernel（補斷裂）
OPEN_ITERS  = 1
CLOSE_ITERS = 2

# 視覺抖動平滑
SMOOTH_WIN = 60
OVERLAY_ALPHA = 0.35
SMOOTH_ALPHA = 0.20      # EMA 平滑係數（越小越平滑，建議 0.18~0.30）

# --- Anti-divergence (防飄移) 參數 ---
ANGLE_STEP_DEG_MAX = 12   # 允許每幀角度變化（度）
XBTM_STEP_MAX      = 60   # 允許底部交點位移（像素）
WIDTH_SHRINK_MIN   = 0.70 # 新寬度不得小於歷史均值的 70%
WIDTH_GROW_MAX     = 1.25 # 新寬度不得大於歷史均值的 125%

# --- 右線搜尋楔形（沿用上一幀右邊界） ---
SEARCH_WEDGE_HALF_W_BTM = 45   # 楔形在下端的半寬（像素）
SEARCH_WEDGE_HALF_W_TOP = 80   # 楔形在上端的半寬（像素）
MAX_LR_ANGLE_DIFF_DEG   = 18   # 左右線角度允許差（度）
CENTER_MARGIN_RATIO     = 0.25 # 右線底部需在中心右側 ≥ 此比例 * 參考寬度

# ---------- 工具函式 ----------
def make_line_points(slope, intercept, y1, y2):
    if abs(slope) < 1e-6:
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, int(y1)), (x2, int(y2))

def angle_deg(slope: float) -> float:
    return float(np.degrees(np.arctan(slope)))

def x_at_y(s: float, b: float, y: float) -> float:
    if abs(s) < 1e-6:
        return np.inf
    return (y - b) / s

def width_at_bottom(L, R, yb) -> float:
    # L/R 皆為 (slope, intercept)
    return abs(x_at_y(R[0], R[1], yb) - x_at_y(L[0], L[1], yb))

def line_reliable(line, prev_line, yb, max_dtheta_deg=ANGLE_STEP_DEG_MAX, max_dx=XBTM_STEP_MAX) -> bool:
    """與上一幀相比，角度與底部交點位移都在容許範圍內才算可靠"""
    if line is None:
        return False
    if prev_line is None:
        return True  # 第一幀/無上一幀時，只要有線就先當作可靠
    dtheta = abs(angle_deg(line[0]) - angle_deg(prev_line[0]))
    dx_btm = abs(x_at_y(line[0], line[1], yb) - x_at_y(prev_line[0], prev_line[1], yb))
    return (dtheta <= max_dtheta_deg) and (dx_btm <= max_dx)

def predict_right_from_left(left, prev_right, w_btm_ref, yb):
    """
    用『左線 + 歷史底部寬度』推回右線：
    - 右線斜率沿用上一幀的右線（若沒有，就用 |左線斜率| 確保為正）
    - 右線在 y=yb 的 x 位置 = 左線底部 x + 參考寬度
    - 再反推右線截距
    """
    sL, bL = left
    xLb = x_at_y(sL, bL, yb)
    if prev_right is not None:
        sR = prev_right[0]
        if sR <= 0:
            sR = abs(sR) if abs(sR) > 1e-6 else abs(sL)
    else:
        sR = abs(sL)
    xRb = xLb + w_btm_ref
    bR  = yb - sR * xRb
    return (sR, bR)

def blend_line(prev, cur, alpha=SMOOTH_ALPHA):
    """對 (slope, intercept) 做指數移動平均；prev/cur 允許為 None"""
    if cur is None:
        return prev
    if prev is None:
        return cur
    s = alpha * cur[0] + (1 - alpha) * prev[0]
    b = alpha * cur[1] + (1 - alpha) * prev[1]
    return (float(s), float(b))

def build_edge_wedge(p_bottom, p_top, half_w_btm, half_w_top) -> np.ndarray:
    """
    以右邊界（rb→rt）為中線，沿法向各偏移 ±半寬，建立細長四邊形「楔形區」。
    """
    p_bottom = np.array(p_bottom, dtype=float)
    p_top    = np.array(p_top,    dtype=float)
    v = p_top - p_bottom
    L = np.hypot(v[0], v[1])
    if L < 1e-6:
        return None
    # 與邊界垂直的單位向量（影像座標）
    n = np.array([ v[1], -v[0] ], dtype=float) / L

    p_bl = p_bottom - n * half_w_btm
    p_br = p_bottom + n * half_w_btm
    p_tl = p_top    - n * half_w_top
    p_tr = p_top    + n * half_w_top

    return np.array([p_bl, p_tl, p_tr, p_br], dtype=np.int32)

def point_in_poly(poly: np.ndarray, x: float, y: float) -> bool:
    if poly is None:
        return True
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

def pick_lane_lines(lines, w, h, y_top, y_bottom,
                    right_wedge=None, center_margin=None, thetaL_deg=None):
    """
    強化版：多了 3 個可選條件
      - right_wedge: 右線候選必須在楔形區內
      - center_margin: 右線底部交點需 > center + margin
      - thetaL_deg: 右線角度需和左線接近（±MAX_LR_ANGLE_DIFF_DEG）
    """
    if lines is None:
        return None, None

    center = w / 2.0
    y_gate = int(h * BAND_BOTTOM_RATIO)

    left_cands, right_cands = [], []  # (x_btm, length, slope, intercept)

    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if y1 < y_top and y2 < y_top:
            continue
        if abs(slope) < 0.3:
            continue
        if max(y1, y2) < y_gate:
            continue

        intercept = y1 - slope * x1
        x_btm = (y_bottom - intercept) / slope
        if x_btm < 0 or x_btm >= w:
            continue

        length = float(np.hypot(x2 - x1, y2 - y1))
        theta_deg = angle_deg(slope)

        # 左側（負斜率 + 底部在中心左邊）
        if slope < 0 and x_btm < center:
            left_cands.append((x_btm, length, slope, intercept))
            continue

        # 右側（正斜率 + 底部在中心右邊）
        if slope > 0 and x_btm > center:
            # 三重過濾
            if right_wedge is not None:
                xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0  # 以中點測
                if not point_in_poly(right_wedge, xm, ym):
                    continue
            if center_margin is not None and (x_btm <= center + center_margin):
                continue
            if thetaL_deg is not None and (abs(theta_deg - abs(thetaL_deg)) > MAX_LR_ANGLE_DIFF_DEG):
                continue
            right_cands.append((x_btm, length, slope, intercept))

    # 左：選 x_btm 最大（最靠近中心），並列取長者
    left = None
    if left_cands:
        left_cands.sort(key=lambda t: (-t[0], -t[1]))
        left = (left_cands[0][2], left_cands[0][3])

    # 右：選 x_btm 最小（最靠近中心），並列取長者（已被前述條件篩過）
    right = None
    if right_cands:
        right_cands.sort(key=lambda t: (t[0], -t[1]))
        right = (right_cands[0][2], right_cands[0][3])

    return left, right

def lane_polygon_from_lines(s_left, b_left, s_right, b_right, y_bottom, y_top):
    L = make_line_points(s_left,  b_left,  y_bottom, y_top)
    R = make_line_points(s_right, b_right, y_bottom, y_top)
    if not L or not R:
        return None
    (x_lb, yb), (x_lt, yt) = L
    (x_rb, _ ), (x_rt, _ ) = R
    return np.array([(x_lb, yb), (x_lt, yt), (x_rt, yt), (x_rb, yb)], dtype=np.int32)

def clip_lane_poly_to_y(poly: np.ndarray, y_new_top: int) -> np.ndarray:
    pts = poly.reshape(-1, 2).astype(float)
    x_lb, yb = pts[0]
    x_lt, yt_l = pts[1]
    x_rt, yt_r = pts[2]
    x_rb, _    = pts[3]

    y_top_orig = min(yt_l, yt_r)
    y_new_top = float(np.clip(y_new_top, y_top_orig, yb))

    denom = yt_l - yb
    if abs(denom) < 1e-6:
        denom = -1e-6

    t = (y_new_top - yb) / denom
    x_new_left  = x_lb + t * (x_lt - x_lb)
    x_new_right = x_rb + t * (x_rt - x_rb)

    clipped = np.array([
        (x_lb,        yb),
        (x_new_left,  y_new_top),
        (x_new_right, y_new_top),
        (x_rb,        yb)
    ], dtype=np.int32)
    return clipped

def poly_bottom_top_width(poly: np.ndarray):
    pts = poly.reshape(-1, 2)
    w_bottom = abs(pts[3][0] - pts[0][0])
    w_top    = abs(pts[2][0] - pts[1][0])
    return float(w_bottom), float(w_top)

def draw_connected_lines(h, w, y_top, y_bottom, L, R, thickness=6):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if L is not None:
        pL = make_line_points(L[0], L[1], y_bottom, y_top)
        if pL:
            cv2.line(canvas, pL[0], pL[1], (255, 255, 255), thickness, cv2.LINE_AA)
    if R is not None:
        pR = make_line_points(R[0], R[1], y_bottom, y_top)
        if pR:
            cv2.line(canvas, pR[0], pR[1], (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas

def draw_lane_border_lines(overlay, draw_poly, thickness):
    # 沿著四邊形左右邊畫白線
    pts = draw_poly.reshape(-1, 2)
    lb = (int(pts[0][0]), int(pts[0][1]))  # 左下
    lt = (int(pts[1][0]), int(pts[1][1]))  # 左上
    rt = (int(pts[2][0]), int(pts[2][1]))  # 右上
    rb = (int(pts[3][0]), int(pts[3][1]))  # 右下
    cv2.line(overlay, lb, lt, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.line(overlay, rb, rt, (255, 255, 255), thickness, cv2.LINE_AA)

# ---------- 主流程 ----------
def main():
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise SystemExit(f"❌ Cannot open video: {VIDEO_IN}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise SystemExit("❌ Cannot open VideoWriter")

    debug_writer = None
    if SAVE_DEBUG:
        debug_writer = cv2.VideoWriter(DEBUG_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not debug_writer.isOpened():
            raise SystemExit("❌ Cannot open debug VideoWriter")

    connected_writer = None
    if SAVE_CONNECTED:
        connected_writer = cv2.VideoWriter(CONNECTED_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not connected_writer.isOpened():
            raise SystemExit("❌ Cannot open connected VideoWriter")

    # 狀態/歷史
    y_top, y_bottom = int(h * ROI_TOP_RATIO), h - 1
    last_L = None
    last_R = None
    last_poly = None
    width_hist = deque(maxlen=30)
    badR_count = 0  # 右線不可靠的連續幀數（用來動態放寬楔形）

    # 形態學 kernel
    k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, K_OPEN)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, K_CLOSE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 只保留下半部 ROI
        roi = frame.copy()
        roi[:y_top, :] = 0

        # 邊緣偵測 + 清雜訊
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 1.0)
        edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
        edges_clean = cv2.morphologyEx(edges, cv2.MORPH_OPEN,  k_open,  iterations=OPEN_ITERS)
        edges_clean = cv2.morphologyEx(edges_clean, cv2.MORPH_CLOSE, k_close, iterations=CLOSE_ITERS)

        # --- 右線搜尋楔形（沿上一幀右邊界）、中心距離門檻、左右角度一致性 ---
        width_ref = float(np.mean(width_hist)) if len(width_hist) >= 5 else None
        center_margin = float(CENTER_MARGIN_RATIO * width_ref) if width_ref is not None else None

        # 右線楔形：根據最近可靠度動態放寬
        wedge_scale = 1.0 + min(badR_count, 3) * 0.5  # 1.0, 1.5, 2.0, 2.5
        right_wedge = None
        if last_poly is not None:
            pts = last_poly.reshape(-1, 2)
            rb = (int(pts[3][0]), int(pts[3][1]))  # 右下
            rt = (int(pts[2][0]), int(pts[2][1]))  # 右上
            half_w_btm = int(SEARCH_WEDGE_HALF_W_BTM * wedge_scale)
            half_w_top = int(SEARCH_WEDGE_HALF_W_TOP * wedge_scale)
            right_wedge = build_edge_wedge(rb, rt, half_w_btm, half_w_top)

        thetaL_deg = angle_deg(last_L[0]) if last_L is not None else None

        # 偵測線段
        lines = cv2.HoughLinesP(
            edges_clean,
            HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LINE_LENGTH,
            maxLineGap=HOUGH_MAX_LINE_GAP
        )

        # 1) 帶楔形/門檻挑左右主線
        L, R = pick_lane_lines(
            lines, w, h, y_top, y_bottom,
            right_wedge=right_wedge,
            center_margin=center_margin,
            thetaL_deg=thetaL_deg
        )

        # 2) 可靠度判斷 + 右線可靠度計數
        okL = line_reliable(L, last_L, y_bottom)
        okR = line_reliable(R, last_R, y_bottom)
        if okR:
            badR_count = max(0, badR_count - 1)
        else:
            badR_count = min(3, badR_count + 1)

        # 3) 取當前可用線（抓不到就沿用上一幀）
        active_L = L if okL else last_L
        active_R = R if okR else last_R

        # 4) 如果左線可靠、右線不可靠 → 用左線 + 歷史寬度推回右線
        if (active_L is not None) and (active_R is None or not okR):
            if width_ref is not None:
                active_R = predict_right_from_left(active_L, last_R, width_ref, y_bottom)

        # 5) 若兩邊都有，檢查「當幀寬度」是否偏離歷史太多
        if (active_L is not None) and (active_R is not None):
            w_new = width_at_bottom(active_L, active_R, y_bottom)
            if width_ref is not None:
                if not (WIDTH_SHRINK_MIN * width_ref <= w_new <= WIDTH_GROW_MAX * width_ref):
                    # 超界：優先保留左線，重算右線以貼合歷史寬度
                    active_R = predict_right_from_left(active_L, last_R, width_ref, y_bottom)
                    w_new = width_at_bottom(active_L, active_R, y_bottom)
        else:
            w_new = None

        # 6) EMA 平滑（影像顯示/狀態更新都用平滑後的線）
        disp_L = blend_line(last_L, active_L) if active_L is not None else last_L
        disp_R = blend_line(last_R, active_R) if active_R is not None else last_R
        last_L, last_R = disp_L, disp_R

        if (disp_L is not None) and (disp_R is not None) and (w_new is not None):
            width_hist.append(w_new)

        # 7) 準備顯示圖層
        overlay = frame.copy()
        bw_vis  = cv2.cvtColor(edges_clean, cv2.COLOR_GRAY2BGR)

        # 8) 存「連成主線」的黑白影片（用平滑後的線）
        connected_vis = draw_connected_lines(h, w, y_top, y_bottom, disp_L, disp_R, LINE_DRAW_THICKNESS)
        if SAVE_CONNECTED and connected_writer is not None:
            connected_writer.write(connected_vis)

        # 9) 嘗試組當幀四邊形（用平滑後的線）
        poly_now = None
        if disp_L is not None and disp_R is not None:
            poly_now = lane_polygon_from_lines(
                disp_L[0], disp_L[1], disp_R[0], disp_R[1], y_bottom, y_top
            )

        # 10) 防爆條件，決定是否更新 last_poly
        if poly_now is not None:
            accept = True
            new_bot, new_top = poly_bottom_top_width(poly_now)

            if REQUIRE_TOP_SMALLER and new_top >= new_bot:
                accept = False
            if ENFORCE_TOP_X2_LT_BOTTOM and (new_top * TOP_X_MULTIPLIER >= new_bot):
                accept = False
            if last_poly is not None:
                last_bot, _ = poly_bottom_top_width(last_poly)
                if last_bot > 0 and new_bot > last_bot * MAX_BOTTOM_GROWTH:
                    accept = False

            if accept:
                last_poly = poly_now

        # 11) 以 last_poly 為準，裁到 GREEN_TOP_RATIO 後鋪綠並畫白線
        if last_poly is not None:
            draw_top = int(h * GREEN_TOP_RATIO)
            draw_poly = clip_lane_poly_to_y(last_poly, draw_top)
            mask = np.zeros_like(overlay)
            cv2.fillPoly(mask, [draw_poly], (0, 255, 0))
            cv2.addWeighted(mask, OVERLAY_ALPHA, overlay, 1 - OVERLAY_ALPHA, 0, overlay)
            draw_lane_border_lines(overlay, draw_poly, WHITE_LINE_THICKNESS)

        # 12) 寫檔
        writer.write(overlay)
        if SAVE_DEBUG and debug_writer is not None:
            debug_writer.write(bw_vis)

        # 13) 預覽
        if SHOW_PREVIEW:
            cv2.imshow("lane",        cv2.resize(overlay, (960, 540)))
            cv2.imshow("lines_debug", cv2.resize(bw_vis,  (960, 540)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 收尾
    cap.release()
    writer.release()
    if connected_writer is not None:
        connected_writer.release()
    if debug_writer is not None:
        debug_writer.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()
    print(f"✅ Done. Saved to {VIDEO_OUT}")

if __name__ == "__main__":
    main()
