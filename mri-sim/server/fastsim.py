"""Fast k-space simulator.

Each tissue's band-limited k-space is computed once from an oversampled label
map, then every phase-encode line is weighted by that tissue's signal at the
echo the line is acquired on. Fast spin echo echo amplitudes come from an
extended phase graph (EPG) CPMG simulation, so refocusing angles below 180
degrees, T1 recovery and T2 decay along the echo train (blurring) are all
represented. Fat is shifted along the read direction by the chemical shift.
"""
import numpy as np
from . import phantoms as ph

FAT_PPM_HZ_PER_T = 149.0  # 3.5 ppm x 42.58 MHz/T


def is_tse(seq): return seq in ("TSE", "TSE3D", "SE")
def is_gre(seq): return seq in ("GRE2D", "GRE3D")


def esp_ms(p):
    """Echo spacing estimate, same rule as PROAQMR (readout + 4 ms)."""
    return 1000.0 / p["BW"] + 4.0


def line_order(np_lines, etl, ete):
    """Linear, shifted ordering. Returns (echo index per line, shots).

    np_lines phase-encode lines are split into etl contiguous segments; the
    segment that holds the k-space centre is played on echo `ete` (0-based).
    shots[s][e] is the line index acquired on echo e of shot s, or None.
    """
    nshots = int(np.ceil(np_lines / etl))
    jc = (np_lines // 2) // nshots
    echo_of_seg = [(j - jc + ete) % etl for j in range(etl)]
    echo_of_line = np.zeros(np_lines, dtype=int)
    shots = [[None] * etl for _ in range(nshots)]
    for j in range(etl):
        for s in range(nshots):
            ln = j * nshots + s
            if ln < np_lines:
                echo_of_line[ln] = echo_of_seg[j]
                shots[s][echo_of_seg[j]] = ln
    return echo_of_line, shots


def _rf(alpha, phi):
    c2, s2, s = np.cos(alpha / 2) ** 2, np.sin(alpha / 2) ** 2, np.sin(alpha)
    return np.array([
        [c2, np.exp(2j * phi) * s2, -1j * np.exp(1j * phi) * s],
        [np.exp(-2j * phi) * s2, c2, 1j * np.exp(-1j * phi) * s],
        [-0.5j * np.exp(-1j * phi) * s, 0.5j * np.exp(1j * phi) * s, np.cos(alpha)],
    ])


def epg_cpmg(alpha_deg, etl, esp, t1, t2):
    """Echo magnitudes (relative to M0) of a CPMG train after a 90 degree pulse (Weigel EPG)."""
    n = 2 * etl + 4
    F = np.zeros((3, n), dtype=complex)
    F[2, 0] = 1.0
    F = _rf(np.pi / 2, 0.0) @ F
    T = _rf(np.deg2rad(alpha_deg), np.pi / 2)
    e1, e2 = np.exp(-esp / 2 / t1), np.exp(-esp / 2 / t2)

    def relax(F):
        F = F.copy(); F[0:2] *= e2; F[2] *= e1; F[2, 0] += 1 - e1
        return F

    def grad(F):
        Fp = np.concatenate(([0], F[0, :-1]))
        Fm = np.concatenate((F[1, 1:], [0]))
        Fp[0] = np.conj(Fm[0])
        return np.vstack((Fp, Fm, F[2]))

    echoes = []
    for _ in range(etl):
        F = grad(relax(F))
        F = T @ F
        F = grad(relax(F))
        echoes.append(abs(F[0, 0]))
    return np.array(echoes)


def tissue_weights(p, b0, ete, etl, tr_eff):
    """Signal per tissue per echo: dict tissue -> array(etl)."""
    w = {}
    for name in ph.TISSUE_NAMES:
        t1, t2, pd, t2s = ph.params(name, b0)
        if pd == 0:
            w[name] = np.zeros(etl)
            continue
        if is_gre(p["seq"]):
            a = np.deg2rad(p["flip"])
            e1 = np.exp(-p["TR"] / t1)
            sig = pd * np.sin(a) * (1 - e1) / (1 - np.cos(a) * e1) * np.exp(-p["TE"] / t2s)
            w[name] = np.full(etl, sig)
        else:
            esp = p["TE"] if p["seq"] == "SE" else esp_ms(p)
            refoc = 180.0 if p["seq"] == "SE" else p["refoc"]
            rec = 1 - np.exp(-max(tr_eff, 1.0) / t1)
            w[name] = pd * rec * epg_cpmg(refoc, etl, esp, t1, t2)
    return w


def simulate(p, region, fatsat=False, os_factor=3, ideal=False):
    """Noiseless magnitude image (ny=Np, nx=Nr) in arbitrary units x voxel volume.

    Returns dict with image, label map at target resolution and metadata.
    """
    b0 = p["b0"]
    nr, npl = int(p["Nr"]), int(p["Np"])
    fx, fy = p["fovR"], p["fovP"]
    etl = 1 if p["seq"] in ("SE", "GRE2D", "GRE3D") else int(p["ETL"])
    if is_tse(p["seq"]) and p["seq"] != "SE":
        esp = esp_ms(p)
        ete = int(np.clip(round(p["TE"] / esp) - 1, 0, etl - 1))
        tr_eff = p["TR"] - etl * esp
    else:
        esp, ete, tr_eff = 0.0, 0, p["TR"]
    echo_of_line, _ = line_order(npl, etl, ete)
    W = tissue_weights(p, b0, ete, etl, tr_eff)

    fine = ph.render(region, fx, fy, nr * os_factor, npl * os_factor)
    target = ph.render(region, fx, fy, nr, npl)
    kx = np.arange(nr) - nr // 2
    cs_px = FAT_PPM_HZ_PER_T * b0 / p["BW"]
    K = np.zeros((npl, nr), dtype=complex)
    present = np.unique(fine)
    oy, ox = (npl * os_factor - npl) // 2, (nr * os_factor - nr) // 2
    for tid in present:
        name = ph.TISSUE_NAMES[tid]
        wt = W[name]
        if not np.any(wt):
            continue
        m = (fine == tid).astype(float)
        km = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(m))) / os_factor ** 2
        km = km[oy:oy + npl, ox:ox + nr]
        if name in ph.FAT_LIKE:
            km = km * np.exp(-2j * np.pi * kx[None, :] * cs_px / nr)
            if fatsat:
                km = km * 0.08
        per_line = wt[ete] * np.ones(npl) if ideal else wt[echo_of_line]
        K += per_line[:, None] * km
    img = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(K))))
    vox = (fx / nr) * (fy / npl) * p["thk"]
    img = img * vox * b0
    return {"image": img, "K": K, "labels": target, "esp": esp, "te_actual": (ete + 1) * esp if esp else p["TE"],
            "cs_px": cs_px, "echo_weights_roi": W[ph.REGIONS[region]["roi"]][echo_of_line], "kspace_echo": echo_of_line}


def noise_sigma(p, sigma0, gk):
    g = 1 + gk * (p["R"] - 1) ** 2
    return sigma0 * np.sqrt(p["BW"] / (p["Np"] * p["NEX"])) * g * np.sqrt(p["R"])


def roi_mean(img, labels, roi):
    m = labels == ph.TID[roi]
    # erode by one pixel to avoid partial volume at edges
    e = m.copy()
    e[1:, :] &= m[:-1, :]; e[:-1, :] &= m[1:, :]; e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]
    if e.sum() < 4:
        e = m
    return float(img[e].mean()) if e.any() else float("nan")


def psf_fwhm_px(weights):
    """FWHM (pixels) of the phase-direction point-spread function from k-space weighting."""
    n = len(weights)
    big = 16 * n
    k = np.zeros(big)
    k[(big - n) // 2:(big - n) // 2 + n] = weights / max(weights.max(), 1e-12)
    psf = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(k))))
    psf /= psf.max()
    above = np.where(psf >= 0.5)[0]
    return (above[-1] - above[0] + 1) / 16.0
