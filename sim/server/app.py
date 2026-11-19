"""PROAQMR image simulation API.

POST /simulate compares a current and an optimized protocol on a body phantom.
engine="fast" uses the k-space/EPG model (seconds, CPU). engine="koma" runs
the full Bloch simulation in KomaMRI (needs the Julia worker on port 8011).
"""
import base64, io, os, tempfile, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from . import phantoms as ph
from . import fastsim as fs

KOMA_URL = os.environ.get("KOMA_URL", "http://127.0.0.1:" + os.environ.get("KOMA_PORT", "8011"))
WORK = os.environ.get("PROAQMR_WORK", tempfile.gettempdir())
ALLOWED = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]
API_KEY = os.environ.get("PROAQMR_API_KEY", "")

app = FastAPI(title="PROAQMR simulation")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED, allow_methods=["*"], allow_headers=["*"])


class Protocol(BaseModel):
    seq: str = "TSE"
    b0: float = 1.5
    TR: float
    TE: float
    flip: float = 90
    refoc: float = 180
    ETL: int = 1
    fovR: float
    fovP: float
    Nr: int
    Np: int
    thk: float
    NEX: float = 1
    BW: float
    R: float = 1


class SimRequest(BaseModel):
    current: Protocol
    optimized: Protocol
    region: str = "abdomen"
    engine: str = "fast"
    fatsat: bool = False
    baseline_snr: float = 30.0
    gk: float = 0.08
    seed: int = 7
    key: str = ""


def _check_key(k):
    if API_KEY and k != API_KEY:
        raise HTTPException(401, "Wrong or missing access key.")


@app.get("/health")
def health():
    koma = None
    try:
        r = httpx.get(KOMA_URL + "/health", timeout=15)
        koma = r.json()
        if "regions" in koma:  # something other than the KomaMRI worker answered on that port
            koma = {"ok": False, "error": f"{KOMA_URL} is answered by another service, not the KomaMRI worker"}
    except Exception:
        koma = {"ok": False, "error": "KomaMRI worker not running yet (it may still be compiling; see koma.log)"}
    return {"ok": True, "koma": koma, "regions": {k: v["label"] for k, v in ph.REGIONS.items()}}


def _limits(p):
    p = dict(p)
    p["Nr"] = int(min(max(p["Nr"], 32), 512))
    p["Np"] = int(min(max(p["Np"], 32), 512))
    p["ETL"] = int(min(max(p.get("ETL", 1), 1), 256))
    if p["seq"] == "TSE3D":
        p["seq"] = "TSE"   # simulated as a single 2D partition
    if p["seq"] == "GRE3D":
        p["seq"] = "GRE2D"
    return p


def _png(img, lo, hi, out_w, out_h):
    a = np.clip((img - lo) / max(hi - lo, 1e-12), 0, 1)
    im = Image.fromarray((a * 255).astype(np.uint8)).resize((out_w, out_h), Image.BICUBIC)
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _on_canvas(img, p, ref, px_per_mm):
    """Place an image on the reference protocol's physical canvas (mm-true)."""
    w = int(round(p["fovR"] * px_per_mm)); h = int(round(p["fovP"] * px_per_mm))
    im = np.array(Image.fromarray(img.astype(np.float32)).resize((w, h), Image.BICUBIC))
    W = int(round(ref["fovR"] * px_per_mm)); H = int(round(ref["fovP"] * px_per_mm))
    out = np.zeros((H, W), dtype=np.float32)
    ys, xs = (H - h) // 2, (W - w) // 2
    sy0, sx0 = max(0, -ys), max(0, -xs)
    dy0, dx0 = max(0, ys), max(0, xs)
    hh, ww = min(h - sy0, H - dy0), min(w - sx0, W - dx0)
    out[dy0:dy0 + hh, dx0:dx0 + ww] = im[sy0:sy0 + hh, sx0:sx0 + ww]
    return out


def _koma_image(p, region, fatsat, fast_noiseless):
    """Run KomaMRI for protocol p and return a magnitude image (Np, Nr)."""
    from .seqgen import build
    if fs.is_tse(p["seq"]) and p["seq"] != "SE":
        p = dict(p, esp_ms=fs.esp_ms(p))
    seq, adc_lines, info = build(p)
    ok, rep = seq.check_timing()
    if not ok:
        raise HTTPException(422, f"Sequence timing check failed: {rep[:1]}")
    tag = os.path.join(WORK, "proaqmr_" + uuid.uuid4().hex[:10])
    seq.write(tag + ".seq")
    write_phantom(tag + ".phantom", region, p, fatsat)
    t0 = time.time()
    try:
        r = httpx.post(KOMA_URL + "/simulate", json={"seq": tag + ".seq", "phantom": tag + ".phantom",
                                                     "out": tag + ".out", "b0": p["b0"]}, timeout=1800)
    except httpx.ConnectError:
        raise HTTPException(503, "KomaMRI worker is not running yet (it may still be compiling; check koma.log).")
    try:
        js = r.json()
    except ValueError:
        js = {}
    if not js.get("ok"):
        raise HTTPException(500, f"KomaMRI error (HTTP {r.status_code}): " + str(js.get("error") or js.get("detail") or r.text[:300]))
    raw = np.fromfile(tag + ".out", dtype=np.float32)
    sig = raw[0::2] + 1j * raw[1::2]
    img = reorder_recon(sig, adc_lines, int(p["Np"]), int(p["Nr"]))
    img = align_to(img, fast_noiseless)
    for ext in (".seq", ".phantom", ".out"):
        try: os.remove(tag + ext)
        except OSError: pass
    return img, {"koma_seconds": js.get("seconds", time.time() - t0), "gpu": js.get("gpu"), **{k: float(v) for k, v in info.items()}}


def reorder_recon(sig, adc_lines, npl, nr):
    need = len(adc_lines) * nr
    if sig.size < need:
        raise HTTPException(500, f"KomaMRI returned {sig.size} samples, expected {need}.")
    sig = sig[:need].reshape(len(adc_lines), nr)
    K = np.zeros((npl, nr), dtype=complex)
    for i, ln in enumerate(adc_lines):
        K[ln] = sig[i]
    return np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(K))))


def align_to(img, ref):
    """Pick the flip that best matches the fast-engine image (orientation conventions differ)."""
    best, bc = img, -2
    for cand in (img, img[::-1, :], img[:, ::-1], img[::-1, ::-1]):
        c = np.corrcoef(cand.ravel(), ref.ravel())[0, 1]
        if c > bc:
            best, bc = cand, c
    return best


def write_phantom(path, region, p, fatsat, z_reps=None, seed=11):
    """Spins (metres, seconds, rad/s) as Float32: n, x, y, z, rho, T1, T2, T2*, dw.

    Spins are jittered in-plane and spread randomly through the slice thickness
    (z_reps copies per in-plane cell; default 10, set PROAQMR_ZREPS). Crusher and spoiler gradients only remove
    unwanted echo pathways if spins are distributed across the slice; with all
    spins at z = 0 those pathways survive and show up as stripes.
    """
    if z_reps is None:
        z_reps = int(os.environ.get("PROAQMR_ZREPS", "10"))  # more = smoother, slower
    rng = np.random.default_rng(seed)
    ex, ey = ph.REGIONS[region]["extent"]
    pix = min(p["fovR"] / p["Nr"], p["fovP"] / p["Np"])
    step = max(pix / 2.0, 0.6)
    nx, ny = int(ex / step), int(ey / step)
    lab = ph.render(region, nx * step, ny * step, nx, ny)
    xs = (np.arange(nx) - nx / 2 + 0.5) * step
    ys = (np.arange(ny) - ny / 2 + 0.5) * step
    X, Y = np.meshgrid(xs, ys)
    thk = p["thk"]
    cols = {k: [] for k in ("x", "y", "z", "rho", "t1", "t2", "t2s", "dw")}
    dw_fat = -2 * np.pi * fs.FAT_PPM_HZ_PER_T * p["b0"]
    for tid in np.unique(lab):
        name = ph.TISSUE_NAMES[tid]
        t1, t2, pd, t2s = ph.params(name, p["b0"])
        if pd == 0:
            continue
        m = lab == tid
        if name in ph.FAT_LIKE and fatsat:
            pd *= 0.08
        bx, by = X[m], Y[m]
        n = bx.size * z_reps
        x = np.repeat(bx, z_reps) + rng.uniform(-step / 2, step / 2, n)
        y = np.repeat(by, z_reps) + rng.uniform(-step / 2, step / 2, n)
        z = rng.uniform(-thk / 2, thk / 2, n)
        cols["x"].append(x * 1e-3); cols["y"].append(y * 1e-3); cols["z"].append(z * 1e-3)
        cols["rho"].append(np.full(n, pd / z_reps)); cols["t1"].append(np.full(n, t1 * 1e-3))
        cols["t2"].append(np.full(n, t2 * 1e-3)); cols["t2s"].append(np.full(n, t2s * 1e-3))
        cols["dw"].append(np.full(n, dw_fat if name in ph.FAT_LIKE else 0.0))
    arrs = [np.concatenate(cols[k]).astype(np.float32) for k in ("x", "y", "z", "rho", "t1", "t2", "t2s", "dw")]
    n = arrs[0].size
    np.concatenate([np.array([n], dtype=np.float32)] + arrs).tofile(path)
    return n


@app.post("/simulate")
def simulate(req: SimRequest):
    _check_key(req.key)
    if req.region not in ph.REGIONS:
        raise HTTPException(400, "Unknown region.")
    t0 = time.time()
    roi = ph.REGIONS[req.region]["roi"]
    pc, po = _limits(req.current.model_dump()), _limits(req.optimized.model_dump())
    out = {"region": ph.REGIONS[req.region]["label"], "roi": roi, "engine": req.engine}
    res = {}
    for tag, p in (("current", pc), ("optimized", po)):
        f = fs.simulate(p, req.region, fatsat=req.fatsat)
        ideal = fs.simulate(p, req.region, fatsat=req.fatsat, ideal=True)
        res[tag] = {"p": p, "fast": f, "ideal": ideal}
    # noise level fixed by the current protocol's SNR in the ROI
    s_cur = fs.roi_mean(res["current"]["fast"]["image"], res["current"]["fast"]["labels"], roi)
    sig_cur = fs.noise_sigma(pc, 1.0, req.gk)
    sigma0 = s_cur / req.baseline_snr / sig_cur
    rng = np.random.default_rng(req.seed)
    meta = {}
    if req.engine == "koma":
        kc, mc = _koma_image(pc, req.region, req.fatsat, res["current"]["fast"]["image"])
        ko, mo = _koma_image(po, req.region, req.fatsat, res["optimized"]["fast"]["image"])
        # put KomaMRI intensities on the fast engine's scale (one factor for both, thickness-weighted)
        lab_c = res["current"]["fast"]["labels"]
        kscale = s_cur / max(fs.roi_mean(kc * pc["thk"], lab_c, roi), 1e-12)
        res["current"]["img"] = kc * pc["thk"] * kscale
        res["optimized"]["img"] = ko * po["thk"] * kscale
        meta = {"current": mc, "optimized": mo}
    else:
        for tag in res:
            res[tag]["img"] = res[tag]["fast"]["image"]
    px_per_mm = 480.0 / pc["fovR"]
    canv = {}
    for tag, r in res.items():
        p = r["p"]
        sigma = fs.noise_sigma(p, sigma0, req.gk)
        noisy = np.abs(r["img"] + rng.normal(0, sigma, r["img"].shape) + 1j * rng.normal(0, sigma, r["img"].shape))
        lab = r["fast"]["labels"]
        s = fs.roi_mean(r["img"], lab, roi)
        fwhm = fs.psf_fwhm_px(r["fast"]["echo_weights_roi"]) if fs.is_tse(p["seq"]) else 1.207
        r["metrics"] = {
            "snr_roi": s / sigma,
            "signal_roi": s,
            "noise_sigma": sigma,
            "psf_fwhm_px": fwhm,
            "eff_phase_res_mm": fwhm * p["fovP"] / p["Np"],
            "nominal_phase_res_mm": p["fovP"] / p["Np"],
            "read_res_mm": p["fovR"] / p["Nr"],
            "chem_shift_px": r["fast"]["cs_px"],
            "chem_shift_mm": r["fast"]["cs_px"] * p["fovR"] / p["Nr"],
            "esp_ms": r["fast"]["esp"],
            "te_actual_ms": r["fast"]["te_actual"],
        }
        canv[tag] = {"noisy": _on_canvas(noisy, p, pc, px_per_mm), "clean": _on_canvas(r["img"], p, pc, px_per_mm)}
    hi = float(np.percentile(canv["current"]["noisy"], 99.5))
    H, W = canv["current"]["noisy"].shape
    diff = canv["optimized"]["clean"] - canv["current"]["clean"]
    dmax = float(np.percentile(np.abs(diff), 99.5)) or 1.0
    out["images"] = {
        "current": _png(canv["current"]["noisy"], 0, hi, W, H),
        "optimized": _png(canv["optimized"]["noisy"], 0, hi, W, H),
        "difference": _png(diff, -dmax, dmax, W, H),
    }
    out["metrics"] = {tag: res[tag]["metrics"] for tag in res}
    out["metrics"]["snr_ratio_pct"] = 100 * res["optimized"]["metrics"]["snr_roi"] / res["current"]["metrics"]["snr_roi"]
    out["koma"] = meta
    out["seconds"] = time.time() - t0
    out["notes"] = [
        "Single 2D slice through the phantom; 3D sequences are simulated as one partition.",
        "Parallel imaging is applied as a noise penalty (g-factor and square root of R); unfolding artifacts are not simulated.",
        "Inversion recovery, flow, motion and vendor deep-learning reconstruction are not simulated.",
    ]
    return out


# ---- Job queue: RunPod's proxy closes requests after about 100 s, so long
# KomaMRI runs are submitted as jobs and polled.
JOBS = {}
_pool = ThreadPoolExecutor(max_workers=1)
_lock = threading.Lock()


def _run_job(jid, req):
    with _lock:
        JOBS[jid]["status"] = "running"; JOBS[jid]["started"] = time.time()
    try:
        result = simulate(req)
        with _lock:
            JOBS[jid].update(status="done", result=result)
    except HTTPException as e:
        with _lock:
            JOBS[jid].update(status="error", error=str(e.detail))
    except Exception as e:  # noqa: BLE001
        with _lock:
            JOBS[jid].update(status="error", error=f"{type(e).__name__}: {e}")


@app.post("/jobs")
def submit(req: SimRequest):
    _check_key(req.key)
    jid = uuid.uuid4().hex[:12]
    with _lock:
        # keep the 50 most recent jobs
        for old in sorted(JOBS, key=lambda k: JOBS[k]["created"])[:-49]:
            JOBS.pop(old, None)
        queued = sum(1 for j in JOBS.values() if j["status"] in ("queued", "running"))
        JOBS[jid] = {"status": "queued", "created": time.time()}
    _pool.submit(_run_job, jid, req)
    return {"id": jid, "queued_ahead": queued}


@app.get("/jobs/{jid}")
def job(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "Unknown job.")
    out = {"status": j["status"]}
    if j["status"] == "done":
        out["result"] = j["result"]
    if j["status"] == "error":
        out["error"] = j["error"]
    if "started" in j:
        out["elapsed"] = time.time() - j["started"]
    return out
