
import shutil, subprocess
from pathlib import Path
import cv2, numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from tqdm import tqdm

cv2.setUseOptimized(True)

FFMPEG_PATH = shutil.which("ffmpeg")
CRF         = 18  

STOP_TH               = 45     
MIN_ACTIVE_FRAMES     = 150    
WARMUP_GRACE_FRAMES   = 150    
REQUIRE_FIRST_PRESENCE= True   
START_TH              = 2      
SAVE_PREVIEW_OVERLAY  = True   
PREVIEW_SUFFIX        = "_preview"
DRAW_ALWAYS           = True

BP_THRESH             = 105    
ROI_ENTER_FRAC        = 0.020  
ROI_KEEP_FRAC         = 0.015  
ROI_BLOB_MIN_FRAC     = 0.015  
ROI_BLOB_MAX_FRAC     = 0.65   
MORPH_OPEN            = 7
MORPH_CLOSE           = 11

GRAY_USE_CLAHE        = True   
CLAHE_CLIP            = 2.0
CLAHE_TILE            = (8,8)

EDGE_SIGMA_AUTO       = True   
CANNY_LOW             = 50     
CANNY_HIGH            = 150

EDGE_FRAC_ENTER       = 0.08   
EDGE_FRAC_KEEP        = 0.06   
EDGE_BLOB_MIN_FRAC    = 0.010  
GRAD_ALPHA            = 0.25   
GRAD_REL_ENTER        = 0.95   
GRAD_REL_KEEP         = 0.85   
LOW_CHROMA_STD        = 6.5    

def make_kernel(k):
    k = int(k)
    if k < 3: return None
    if k % 2 == 0: k += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

def clamp_rect(x, y, w, h, W, H):
    x = max(0, min(int(x), W-1))
    y = max(0, min(int(y), H-1))
    w = max(2, min(int(w), W-x))
    h = max(2, min(int(h), H-y))
    return x, y, w, h

def make_even_rect(x, y, w, h):
    if w % 2: w -= 1
    if h % 2: h -= 1
    return x, y, max(2, w), max(2, h)

def ycc_chroma_std(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    _, Cr, Cb = cv2.split(ycc)
    return float(Cr.std()), float(Cb.std())

def build_skin_hist(bgr_roi):
    ycc = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2YCrCb)
    hist = cv2.calcHist([ycc], [1, 2], None, [128, 128], [0,256, 0,256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist

def skin_backproj(bgr, hist):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    bp  = cv2.calcBackProject([ycc], [1, 2], hist, [0,256, 0,256], scale=1)
    return cv2.GaussianBlur(bp, (7,7), 0)

def present_color(bgr_roi, skin_hist, roi_area, k_open, k_close):
    bp = skin_backproj(bgr_roi, skin_hist)
    _, binm = cv2.threshold(bp, BP_THRESH, 255, cv2.THRESH_BINARY)
    if k_open  is not None: binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN,  k_open)
    if k_close is not None: binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k_close)

    total_on = int(cv2.countNonZero(binm))
    skin_frac = total_on / max(1.0, roi_area)

    contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    has_big_blob = False
    if contours:
        amax = max(cv2.contourArea(c) for c in contours)
        has_big_blob = (amax >= ROI_BLOB_MIN_FRAC*roi_area) and (amax <= ROI_BLOB_MAX_FRAC*roi_area)

    enough_now  = (skin_frac >= ROI_ENTER_FRAC)
    enough_keep = (skin_frac >= ROI_KEEP_FRAC)

    return skin_frac, has_big_blob and (enough_now or enough_keep)

def clahe_gray(gray):
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    return clahe.apply(gray)

def canny_auto(gray):
    
    v = np.median(gray)
    low  = int(max(0, (1.0 - 0.33) * v))
    high = int(min(255, (1.0 + 0.33) * v))
    if low >= high:
        low, high = 50, 150
    return cv2.Canny(gray, low, high)

def grad_energy(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)   
    return float(mag.mean())

def present_gray(bgr_roi, roi_area, state):
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    if GRAY_USE_CLAHE:
        gray = clahe_gray(gray)

    edges = canny_auto(gray) if EDGE_SIGMA_AUTO else cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    edge_on = int(cv2.countNonZero(edges))
    edge_frac = edge_on / max(1.0, roi_area)

    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_edge_blob = 0.0
    if contours:
        amax = max(cv2.contourArea(c) for c in contours)
        max_edge_blob = amax / max(1.0, roi_area)

    
    ge = grad_energy(gray)
    if state["grad_base"] is None:
        state["grad_base"] = ge
        state["grad_ema"]  = ge
    state["grad_ema"] = GRAD_ALPHA*ge + (1.0-GRAD_ALPHA)*state["grad_ema"]
    rel = state["grad_ema"] / max(1e-6, state["grad_base"])

    
    enough_now  = (edge_frac >= EDGE_FRAC_ENTER) or (rel >= GRAD_REL_ENTER)
    enough_keep = (edge_frac >= EDGE_FRAC_KEEP)  or (rel >= GRAD_REL_KEEP)

    has_part = (max_edge_blob >= EDGE_BLOB_MIN_FRAC) and (enough_now or enough_keep)
    return (edge_frac, rel, max_edge_blob), has_part


def pick_paths():
    root = tk.Tk(); root.withdraw()
    try:
        root.lift(); root.attributes('-topmost', True); root.after(300, lambda: root.attributes('-topmost', False))
    except Exception: pass
    in_file = filedialog.askopenfilename(
        title="Válaszd ki a bemeneti videót",
        filetypes=[("Videó","*.mp4;*.avi;*.mov;*.mkv;*.MP4;*.AVI;*.MOV;*.MKV"), ("Minden fájl","*.*")]
    )
    if not in_file: messagebox.showinfo("Kilépés","Nem választottál bemeneti fájlt."); return None, None, None
    out_dir = filedialog.askdirectory(title="Válaszd ki a mentési mappát")
    if not out_dir: messagebox.showinfo("Kilépés","Nem választottál mentési mappát."); return None, None, None
    base = simpledialog.askstring("Kimeneti név","Adj egy nevet (kiterjesztés nélkül):", initialvalue="roi_cut")
    if not base: messagebox.showinfo("Kilépés","Nem adtál meg nevet."); return None, None, None
    return Path(in_file), Path(out_dir), base

def open_writer(out_path, size, fps):
    W, H = size
    if FFMPEG_PATH:
        cmd = [FFMPEG_PATH, "-y", "-f","rawvideo","-pix_fmt","bgr24","-s",f"{W}x{H}","-r",str(fps),"-i","-",
               "-c:v","libx264","-preset","slow","-crf",str(CRF),"-pix_fmt","yuv420p", str(out_path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        return ("ffmpeg", proc)
    wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W,H))
    if not wr.isOpened(): raise SystemExit(f"Nem nyitható: {out_path}")
    try: wr.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
    except Exception: pass
    return ("avi", wr)

def main():
    in_path, out_dir, base = pick_paths()
    if not in_path: return

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened(): raise SystemExit(f"Nem sikerült megnyitni: {in_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0

    ok, first = cap.read()
    if not ok or first is None: raise SystemExit("Nem sikerült beolvasni az első képkockát.")
    H, W = first.shape[:2]

    
    cv2.namedWindow("ROI valasztas", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI valasztas", min(W,1280), min(H,720))
    x, y, w, h = cv2.selectROI("ROI valasztas", first, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("ROI valasztas")
    if w == 0 or h == 0: raise SystemExit("Nem jelöltél ki ROI-t.")
    x, y, w, h = clamp_rect(x, y, w, h, W, H)
    x, y, w, h = make_even_rect(x, y, w, h)
    print(f"[INFO] ROI: x={x}, y={y}, w={w}, h={h}")

    roi0 = first[y:y+h, x:x+w]
    cr_std, cb_std = ycc_chroma_std(roi0)
    use_gray = (cr_std <= LOW_CHROMA_STD and cb_std <= LOW_CHROMA_STD)
    print(f"[INFO] Mód: {'GRAY' if use_gray else 'COLOR'}  (Crσ={cr_std:.2f}, Cbσ={cb_std:.2f})")

    
    skin_hist = build_skin_hist(roi0) if not use_gray else None
    gray_state = {"grad_base": None, "grad_ema": None} if use_gray else None
    roi_area   = float(w*h)

    
    crop_out = (out_dir / base).with_suffix(".mp4" if FFMPEG_PATH else ".avi")
    kind_c, wr_c = open_writer(crop_out, (w, h), fps)
    print("[INFO] Croppolt kimenet:", crop_out)

    if SAVE_PREVIEW_OVERLAY:
        prev_out = (out_dir / (base + PREVIEW_SUFFIX)).with_suffix(".mp4" if FFMPEG_PATH else ".avi")
        kind_p, wr_p = open_writer(prev_out, (W, H), fps)
        print("[INFO] Preview overlay:", prev_out)
    else:
        wr_p = None

    k_open  = make_kernel(MORPH_OPEN)
    k_close = make_kernel(MORPH_CLOSE)

    
    present_streak = 0
    absent_streak  = 0
    wrote_frames   = 0
    cut_frame_idx  = None
    armed = (not REQUIRE_FIRST_PRESENCE)

    with tqdm(total=total, desc="Kivágás fut", unit="frame") as pbar:
        idx = 0
        cur = first
        while True:
            if idx > 0:
                ok, cur = cap.read()
                if not ok or cur is None: break

            crop = cur[y:y+h, x:x+w]

            if use_gray:
                (edge_frac, grad_rel, edge_blob), has_part = present_gray(crop, roi_area, gray_state)
                debug_txt = f"edge:{edge_frac:.3f}  grad_rel:{grad_rel:.2f}  blob:{edge_blob:.3f}"
                color = (0,200,0) if has_part else (0,0,255)
            else:
                skin_frac, has_part = present_color(crop, skin_hist, roi_area, k_open, k_close)
                debug_txt = f"skin:{skin_frac:.3f}"
                color = (0,200,0) if has_part else (0,0,255)

            
            if has_part:
                present_streak += 1
                absent_streak   = 0
            else:
                absent_streak  += 1
                present_streak  = 0

           
            if REQUIRE_FIRST_PRESENCE and (not armed) and (present_streak >= START_TH):
                armed = True

            
            if (armed
                and idx >= WARMUP_GRACE_FRAMES
                and wrote_frames >= MIN_ACTIVE_FRAMES
                and absent_streak >= STOP_TH):
                cut_frame_idx = idx
                break

            
            if kind_c == "ffmpeg": wr_c.stdin.write(crop.tobytes())
            else:                  wr_c.write(crop)
            wrote_frames += 1

            
            if wr_p is not None:
                vis = cur.copy()
                if DRAW_ALWAYS:
                    cv2.rectangle(vis, (x,y), (x+w, y+h), color, 2)
                cv2.putText(vis, debug_txt + f"  pres:{present_streak} abs:{absent_streak}/{STOP_TH} armed:{int(armed)}",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
                if kind_p == "ffmpeg": wr_p.stdin.write(vis.tobytes())
                else:                  wr_p.write(vis)

            idx += 1
            pbar.update(1)

    
    cap.release()
    if kind_c == "ffmpeg": wr_c.stdin.close(); wr_c.wait()
    else:                  wr_c.release()
    if wr_p is not None:
        if kind_p == "ffmpeg": wr_p.stdin.close(); wr_p.wait()
        else:                  wr_p.release()

    
    if cut_frame_idx is not None:
        t = (cut_frame_idx / (fps or 25.0))
        print(f"[OK] Robusztus hiány → vágás itt: frame={cut_frame_idx}  t={t:.3f}s  | {crop_out}")
    else:
        print(f"[OK] Nem volt robusztus hiány → teljes ROI mentve: {crop_out}")

if __name__ == "__main__":
    main()
