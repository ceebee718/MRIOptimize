"""Pulseq sequences that follow PROAQMR protocol parameters (single 2D slice).

These are generic physics implementations (CPMG fast spin echo, spin echo,
RF-spoiled gradient echo), not vendor sequences. They are used as KomaMRI
input. Each builder returns (seq, adc_lines, info) where adc_lines lists the
phase-encode line index of every ADC event in playback order.
"""
import numpy as np
import pypulseq as pp
from .fastsim import line_order


def system(b0):
    return pp.Opts(max_grad=32, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s",
                   rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6, B0=b0)


def _round_up(t, raster=1e-5):
    return np.ceil(t / raster - 1e-9) * raster


def tse(p):
    sys = system(p["b0"])
    seq = pp.Sequence(sys)
    nr, npl = int(p["Nr"]), int(p["Np"])
    fov_r, fov_p, thk = p["fovR"] * 1e-3, p["fovP"] * 1e-3, p["thk"] * 1e-3
    etl = 1 if p["seq"] == "SE" else int(p["ETL"])
    refoc = np.deg2rad(180.0 if p["seq"] == "SE" else p["refoc"])
    dkx, dky = 1 / fov_r, 1 / fov_p
    dwell = round(1.0 / (p["BW"] * nr) / 1e-7) * 1e-7  # ADC dwell on 100 ns raster
    t_ro = _round_up(dwell * nr)  # readout flat time on gradient raster

    rf_ex, gz_ex, gz_reph = pp.make_sinc_pulse(flip_angle=np.pi / 2, phase_offset=np.pi / 2, duration=2e-3,
                                               slice_thickness=thk, time_bw_product=4, apodization=0.5,
                                               system=sys, return_gz=True, use="excitation")
    rf_ref, gz_ref, _ = pp.make_sinc_pulse(flip_angle=refoc, phase_offset=0.0, duration=2e-3,
                                           slice_thickness=thk, time_bw_product=4, apodization=0.5,
                                           system=sys, return_gz=True, use="refocusing")
    gx = pp.make_trapezoid("x", flat_area=(dkx / dwell) * t_ro, flat_time=t_ro, system=sys)
    adc = pp.make_adc(num_samples=nr, dwell=dwell, delay=gx.rise_time + _round_up((t_ro - dwell * nr) / 2, 1e-6), system=sys)
    gx_pre = pp.make_trapezoid("x", area=gx.area / 2, system=sys)
    crush = pp.make_trapezoid("z", area=4 / thk, system=sys)
    gy_max = pp.make_trapezoid("y", area=npl / 2 * dky, system=sys)
    t_c = max(pp.calc_duration(crush), pp.calc_duration(gy_max))
    d_ref = pp.calc_duration(rf_ref, gz_ref)
    d_ro = pp.calc_duration(gx, adc)
    esp_min = d_ref + 2 * t_c + d_ro
    d_ex = pp.calc_duration(rf_ex, gz_ex)
    ex_center = rf_ex.delay + pp.calc_rf_center(rf_ex)[0]
    d_b = max(pp.calc_duration(gz_reph), pp.calc_duration(gx_pre))
    esp = max(esp_min, 2 * ((d_ex - ex_center) + d_b + t_c + d_ref / 2))
    esp = max(esp, p.get("esp_ms", 0) * 1e-3)
    if p["seq"] == "SE":
        esp = max(esp, p["TE"] * 1e-3)
    esp = _round_up(esp)
    pad = _round_up((esp - esp_min) / 2) if esp > esp_min else 0.0
    t_c2 = t_c + pad
    esp = d_ref + 2 * t_c2 + d_ro
    fill_b = _round_up(esp / 2 - (d_ex - ex_center) - t_c - d_ref / 2)

    ete = int(np.clip(round(p["TE"] * 1e-3 / esp) - 1, 0, etl - 1))
    _, shots = line_order(npl, etl, ete)
    shot_time = d_ex + fill_b + t_c + etl * esp
    tr = max(p["TR"] * 1e-3, shot_time + 1e-3)
    tr_fill = _round_up(tr - shot_time)

    adc_lines = []
    for s_i, shot in enumerate([shots[0]] + shots):  # first shot is a dummy for steady state
        dummy = s_i == 0
        seq.add_block(rf_ex, gz_ex)
        seq.add_block(gz_reph, gx_pre, pp.make_delay(fill_b))
        seq.add_block(crush)
        for e in range(etl):
            ln = shot[e]
            area = 0.0 if ln is None else (ln - npl // 2) * dky
            gy = pp.make_trapezoid("y", area=area, duration=t_c, system=sys)
            gy_r = pp.make_trapezoid("y", area=-area, duration=t_c, system=sys)
            seq.add_block(rf_ref, gz_ref)
            seq.add_block(gy, crush, pp.make_delay(t_c2))
            if dummy or ln is None:
                seq.add_block(gx)
            else:
                seq.add_block(gx, adc)
                adc_lines.append(ln)
            seq.add_block(gy_r, crush, pp.make_delay(t_c2))
        seq.add_block(pp.make_delay(tr_fill))

    seq.set_definition("FOV", [fov_r, fov_p, thk])
    seq.set_definition("Name", "proaqmr_tse")
    info = {"esp_ms": esp * 1e3, "te_ms": (ete + 1) * esp * 1e3, "tr_ms": tr * 1e3, "shots": len(shots)}
    return seq, adc_lines, info


def gre(p, n_dummy=None):
    sys = system(p["b0"])
    seq = pp.Sequence(sys)
    nr, npl = int(p["Nr"]), int(p["Np"])
    fov_r, fov_p, thk = p["fovR"] * 1e-3, p["fovP"] * 1e-3, p["thk"] * 1e-3
    dkx, dky = 1 / fov_r, 1 / fov_p
    dwell = round(1.0 / (p["BW"] * nr) / 1e-7) * 1e-7  # ADC dwell on 100 ns raster
    t_ro = _round_up(dwell * nr)  # readout flat time on gradient raster
    rf, gz, gz_reph = pp.make_sinc_pulse(flip_angle=np.deg2rad(p["flip"]), duration=1e-3, slice_thickness=thk,
                                         time_bw_product=4, apodization=0.5, system=sys, return_gz=True, use="excitation")
    gx = pp.make_trapezoid("x", flat_area=(dkx / dwell) * t_ro, flat_time=t_ro, system=sys)
    adc = pp.make_adc(num_samples=nr, dwell=dwell, delay=gx.rise_time + _round_up((t_ro - dwell * nr) / 2, 1e-6), system=sys)
    gx_pre = pp.make_trapezoid("x", area=-gx.area / 2, system=sys)
    gy_max = pp.make_trapezoid("y", area=npl / 2 * dky, system=sys)
    t_pre = max(pp.calc_duration(gx_pre), pp.calc_duration(gy_max), pp.calc_duration(gz_reph))
    gx_spoil = pp.make_trapezoid("x", area=2 * nr * dkx, system=sys)
    gz_spoil = pp.make_trapezoid("z", area=4 / thk, system=sys)
    t_sp = max(pp.calc_duration(gx_spoil), pp.calc_duration(gz_spoil), t_pre)
    rf_center = rf.delay + pp.calc_rf_center(rf)[0]
    d_rf = pp.calc_duration(rf, gz)
    te_min = (d_rf - rf_center) + t_pre + gx.rise_time + gx.flat_time / 2
    te = max(p["TE"] * 1e-3, te_min)
    te_fill = _round_up(te - te_min)
    tr_min = d_rf + t_pre + te_fill + pp.calc_duration(gx) + t_sp
    tr = max(p["TR"] * 1e-3, tr_min)
    tr_fill = _round_up(tr - tr_min)

    if n_dummy is None:
        n_dummy = int(min(120, max(10, 5000.0 / max(p["TR"], 1e-3))))
    rf_phase, rf_inc = 0.0, 0.0
    adc_lines = []
    order = [None] * n_dummy + list(range(npl))
    for ln in order:
        rf.phase_offset = np.deg2rad(rf_phase)
        adc.phase_offset = np.deg2rad(rf_phase)
        rf_inc = (rf_inc + 117.0) % 360.0
        rf_phase = (rf_phase + rf_inc) % 360.0
        area = 0.0 if ln is None else (ln - npl // 2) * dky
        gy = pp.make_trapezoid("y", area=area, duration=t_pre, system=sys)
        gy_r = pp.make_trapezoid("y", area=-area, duration=t_sp, system=sys)
        seq.add_block(rf, gz)
        seq.add_block(gx_pre, gy, gz_reph)
        if te_fill > 0:
            seq.add_block(pp.make_delay(te_fill))
        if ln is None:
            seq.add_block(gx)
        else:
            seq.add_block(gx, adc)
            adc_lines.append(ln)
        seq.add_block(gx_spoil, gy_r, gz_spoil)
        if tr_fill > 0:
            seq.add_block(pp.make_delay(tr_fill))
    seq.set_definition("FOV", [fov_r, fov_p, thk])
    seq.set_definition("Name", "proaqmr_gre")
    return seq, adc_lines, {"te_ms": te * 1e3, "tr_ms": tr * 1e3}


def build(p):
    if p["seq"] in ("GRE2D", "GRE3D"):
        return gre(p)
    return tse(p)
