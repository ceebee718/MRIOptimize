# PROAQMR simulation server

Simulates MRI images for a current and an optimized protocol on a body phantom, so a physicist and radiologist can see the trade-off (SNR, echo-train blurring, chemical shift) before a protocol change goes on a scanner. The PROAQMR web page calls this server.

Two engines:

| Engine | What it does | Speed | Needs |
|---|---|---|---|
| `fast` | k-space model. Every phase-encode line is weighted by the tissue signal at the echo it is acquired on; fast spin echo echo amplitudes come from an extended phase graph (refocusing angle, T1, T2). | 1–3 s per comparison | CPU only |
| `koma` | Full Bloch simulation in [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl) of a Pulseq sequence built from the protocol. | seconds to minutes | GPU pod, Julia |

Phantoms: `abdomen` (axial liver, kidneys, spleen, vessels, spine, 4–12 mm liver lesions) and `lumbar` (sagittal vertebrae, discs, thecal sac, conus, a disc protrusion and a marrow lesion). Tissue relaxation values are approximate and meant for relative comparison.

## Deploy on RunPod

1. **Put this folder in a GitHub repo**, for example `proaqmr-sim`. Files go at the repo root (`setup.sh`, `start.sh`, `requirements.txt`, `server/...`).
2. **Create a GPU pod** in RunPod:
   - Template: *RunPod PyTorch* (Ubuntu 22.04, CUDA 12). Any recent NVIDIA GPU works; an RTX A4000/A5000 or L4 is plenty.
   - Volume disk: 20 GB or more, mounted at `/workspace`.
   - **Expose HTTP port `8000`** (Edit Pod → Expose HTTP Ports).
3. **Open the pod's web terminal** (Connect → Start Web Terminal) and run:
   ```bash
   cd /workspace
   git clone https://github.com/YOUR-ACCOUNT/proaqmr-sim.git
   cd proaqmr-sim
   bash setup.sh          # one time, 10–20 minutes (Julia + KomaMRI + CUDA)
   export PROAQMR_API_KEY='choose-a-long-random-string'
   export ALLOWED_ORIGINS='https://YOUR-SITE.netlify.app'
   bash start.sh
   ```
4. **Your server address** is `https://<POD_ID>-8000.proxy.runpod.net` (Connect → HTTP Service [Port 8000]). Check `https://<POD_ID>-8000.proxy.runpod.net/health`; `"koma":{"ok":true,"gpu":true}` means KomaMRI is ready. The first start compiles KomaMRI, so `koma` can take a few minutes to turn `ok`.
5. **In PROAQMR**, open *Image simulation*, paste the server address and the access key, and run.

After a pod restart, only step 3's last three lines are needed (`export ...` and `bash start.sh`); Julia and packages stay in `/workspace`. Stop the pod when you are not using it.

## Check that KomaMRI works

In a second web terminal on the pod, after `start.sh` is running:

```bash
cd /workspace/proaqmr-sim && python3 -m server.selftest http://127.0.0.1:8000 "$PROAQMR_API_KEY"
```

It runs one small comparison on each engine and prints `OK` or the error. If `koma` fails, send the error and the end of `/workspace/koma.log`.

## API

- `GET /health`
- `POST /jobs` with the JSON below → `{"id": ...}`; then poll `GET /jobs/{id}` until `status` is `done` (RunPod's proxy closes single requests after about 100 s, so the page always uses jobs).
- `POST /simulate` runs synchronously (fine for the fast engine).

```json
{
  "current":   {"seq":"TSE","b0":1.5,"TR":4000,"TE":90,"refoc":150,"ETL":18,"fovR":380,"fovP":285,"Nr":320,"Np":224,"thk":6,"NEX":2,"BW":260,"R":2},
  "optimized": {"seq":"TSE","b0":1.5,"TR":3750,"TE":90,"refoc":140,"ETL":36,"fovR":380,"fovP":285,"Nr":320,"Np":216,"thk":6,"NEX":1.5,"BW":160,"R":2},
  "region": "abdomen", "engine": "fast", "fatsat": true, "baseline_snr": 30, "gk": 0.08, "key": "..."
}
```

`seq` is one of `TSE`, `SE`, `GRE2D` (`TSE3D` and `GRE3D` are simulated as one 2D partition). `BW` is Hz per pixel. Times in ms, lengths in mm.

The result has three PNG images (current, optimized, difference, all on the same physical scale and window) and metrics for each protocol: ROI SNR, effective phase-direction resolution from the echo-train point-spread function, chemical shift, echo spacing and actual TE.

## Image quality vs speed (KomaMRI)

Spins are spread randomly through the slice so crusher gradients work; the leftover from unwanted echo pathways shows as fine grain that shrinks with more spins per pixel. `PROAQMR_ZREPS` (default 10) sets the number of spin positions through the slice. Raise it for smoother images, lower it for speed:

```bash
export PROAQMR_ZREPS=20   # before bash start.sh
```

## Limits

- One 2D slice; no slice profile, flow, motion, inversion recovery or vendor deep-learning reconstruction.
- Parallel imaging is a noise penalty (g-factor × √R); unfolding artifacts are not simulated.
- The Pulseq sequences are generic CPMG fast spin echo, spin echo and RF-spoiled gradient echo, not GE/Siemens/Philips implementations.
- The KomaMRI worker (`server/koma_worker.jl`) could not be test-run where it was written. If `/health` shows `koma` not ok, check `/workspace/koma.log`; the fast engine keeps working regardless.
