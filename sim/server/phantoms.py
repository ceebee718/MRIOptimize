"""Analytic body phantoms for PROAQMR image simulation.

Shapes are defined in millimetres around the image centre. x is the
horizontal image axis (frequency / read direction), y is the vertical axis
(phase-encode direction). Shapes are painted in order; later shapes win.
Relaxation values are approximate literature values (ms) and are meant for
relative protocol comparison, not quantitative mapping.
"""
import numpy as np

# name: {field: (T1, T2, PD, T2*)}
TISSUE = {
    "air":       {1.5: (1, 1, 0.0, 1),        3: (1, 1, 0.0, 1)},
    "bone":      {1.5: (1, 1, 0.0, 1),        3: (1, 1, 0.0, 1)},
    "fat":       {1.5: (343, 58, 1.0, 40),    3: (382, 68, 1.0, 35)},
    "muscle":    {1.5: (1008, 44, 0.80, 30),  3: (1412, 50, 0.80, 25)},
    "soft":      {1.5: (900, 60, 0.80, 35),   3: (1200, 55, 0.80, 25)},
    "liver":     {1.5: (576, 46, 0.75, 27),   3: (812, 42, 0.75, 17)},
    "spleen":    {1.5: (1057, 79, 0.85, 50),  3: (1328, 61, 0.85, 40)},
    "kcortex":   {1.5: (966, 87, 0.85, 60),   3: (1142, 76, 0.85, 40)},
    "kmedulla":  {1.5: (1412, 85, 0.90, 60),  3: (1545, 81, 0.90, 40)},
    "marrow":    {1.5: (550, 50, 0.80, 20),   3: (650, 45, 0.80, 12)},
    "fluid":     {1.5: (4000, 2000, 1.0, 1000), 3: (4300, 1800, 1.0, 800)},
    "blood":     {1.5: (1440, 290, 0.90, 200), 3: (1930, 275, 0.90, 150)},
    "nucleus":   {1.5: (1200, 120, 0.95, 60), 3: (1300, 100, 0.95, 50)},
    "annulus":   {1.5: (800, 30, 0.70, 15),   3: (900, 28, 0.70, 12)},
    "cord":      {1.5: (870, 75, 0.85, 50),   3: (1080, 70, 0.85, 40)},
    "hemangioma":{1.5: (1200, 160, 0.95, 80), 3: (1600, 140, 0.95, 60)},
    "lesion":    {1.5: (1000, 80, 0.85, 40),  3: (1300, 70, 0.85, 30)},
}
FAT_LIKE = {"fat"}

# shape: (kind, tissue, params)
#   ellipse: (cx, cy, ax, ay, angle_deg)
#   rect:    (x0, x1, y0, y1)
ABDOMEN = [
    ("ellipse", "fat",      (0, 0, 172, 122, 0)),
    ("ellipse", "muscle",   (0, 0, 160, 110, 0)),
    ("ellipse", "soft",     (0, 2, 150, 100, 0)),
    ("ellipse", "liver",    (-62, -8, 82, 72, -10)),
    ("ellipse", "fluid",    (-45, -58, 14, 9, 20)),       # gallbladder
    ("ellipse", "spleen",   (88, 18, 30, 46, 25)),
    ("ellipse", "kcortex",  (-58, 52, 26, 38, 20)),
    ("ellipse", "kmedulla", (-58, 52, 14, 24, 20)),
    ("ellipse", "kcortex",  (60, 50, 26, 38, -20)),
    ("ellipse", "kmedulla", (60, 50, 14, 24, -20)),
    ("ellipse", "muscle",   (-34, 92, 22, 15, 0)),        # paraspinal
    ("ellipse", "muscle",   (34, 92, 22, 15, 0)),
    ("ellipse", "bone",     (0, 66, 25, 23, 0)),          # vertebral cortex
    ("ellipse", "marrow",   (0, 66, 22, 20, 0)),
    ("ellipse", "fluid",    (0, 96, 9, 8, 0)),            # spinal canal
    ("ellipse", "blood",    (14, 36, 11, 11, 0)),         # aorta
    ("ellipse", "blood",    (-20, 38, 10, 12, 0)),        # IVC
    ("ellipse", "fluid",    (-88, -30, 6, 6, 0)),         # cyst 12 mm
    ("ellipse", "hemangioma", (-38, 10, 4, 4, 0)),        # 8 mm
    ("ellipse", "lesion",   (-74, 26, 3, 3, 0)),          # 6 mm
    ("ellipse", "lesion",   (-100, 5, 2, 2, 0)),          # 4 mm
]

def _lumbar():
    s = [("rect", "soft", (-125, 112, -140, 140)),
         ("rect", "fat", (88, 112, -140, 140)),
         ("rect", "muscle", (48, 88, -140, 140)),
         ("rect", "blood", (-48, -32, -140, 140)),
         ("rect", "fluid", (18, 34, -140, 140)),           # thecal sac
         ("rect", "cord", (23, 29, -140, -35))]            # conus
    centers = [-112, -72, -32, 8, 48, 88]
    for c in centers:
        s += [("rect", "bone", (-22, 17, c - 15, c + 15)),
              ("rect", "marrow", (-20, 15, c - 13, c + 13)),
              ("rect", "bone", (34, 50, c - 10, c + 10)),
              ("rect", "marrow", (36, 48, c - 8, c + 8))]
    for c in centers[:-1]:
        d = c + 20
        s += [("rect", "annulus", (-21, 16, d - 5, d + 5)),
              ("ellipse", "nucleus", (-2, d, 13, 3.2, 0))]
    s += [("ellipse", "nucleus", (19, 48 + 20, 5, 3, 0)),   # posterior disc protrusion
          ("ellipse", "lesion", (-4, 8, 5, 5, 0))]          # marrow lesion
    return s

LUMBAR = _lumbar()

REGIONS = {
    "abdomen": {"label": "Abdomen, axial (liver, kidneys, spleen)", "shapes": ABDOMEN, "roi": "liver", "extent": (344, 244)},
    "lumbar":  {"label": "Lumbar spine, sagittal", "shapes": LUMBAR, "roi": "marrow", "extent": (237, 280)},
}
TISSUE_NAMES = list(TISSUE.keys())
TID = {n: i for i, n in enumerate(TISSUE_NAMES)}


def render(region, fov_x, fov_y, nx, ny):
    """Label map (ny, nx) of tissue ids on a grid covering fov_x by fov_y mm."""
    xs = (np.arange(nx) - nx / 2 + 0.5) * fov_x / nx
    ys = (np.arange(ny) - ny / 2 + 0.5) * fov_y / ny
    X, Y = np.meshgrid(xs, ys)
    lab = np.full((ny, nx), TID["air"], dtype=np.int16)
    for kind, t, p in REGIONS[region]["shapes"]:
        if kind == "ellipse":
            cx, cy, ax, ay, ang = p
            a = np.deg2rad(ang)
            xr = (X - cx) * np.cos(a) + (Y - cy) * np.sin(a)
            yr = -(X - cx) * np.sin(a) + (Y - cy) * np.cos(a)
            m = (xr / ax) ** 2 + (yr / ay) ** 2 <= 1
        else:
            x0, x1, y0, y1 = p
            m = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        lab[m] = TID[t]
    return lab


def params(name, b0):
    f = 3 if b0 >= 2.25 else 1.5
    return TISSUE[name][f]
