"""Quick end-to-end check of both engines. Run on the pod after start.sh:  python3 -m server.selftest"""
import json, sys, time, httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
KEY = sys.argv[2] if len(sys.argv) > 2 else ""
p = dict(seq="TSE", b0=1.5, TR=2000, TE=80, flip=90, refoc=150, ETL=12, fovR=380, fovP=300, Nr=96, Np=96, thk=6, NEX=1, BW=300, R=1)
h = httpx.get(BASE + "/health", timeout=10).json()
print("health:", json.dumps(h))
for engine in ("fast", "koma"):
    t = time.time()
    r = httpx.post(BASE + "/jobs", json=dict(current=p, optimized=dict(p, ETL=16), region="abdomen", engine=engine, key=KEY), timeout=30)
    jid = r.json().get("id")
    if not jid:
        print(f"{engine}: FAILED to submit - {r.status_code} {r.text[:300]}")
        continue
    while True:
        time.sleep(2)
        j = httpx.get(f"{BASE}/jobs/{jid}", timeout=30).json()
        if j.get("status") in ("done", "error"):
            break
        if "status" not in j:
            j = {"status": "error", "error": f"unexpected reply: {str(j)[:300]}"}
            break
    if j["status"] == "done":
        m = j["result"]["metrics"]
        print(f"{engine}: OK in {time.time()-t:.0f} s, SNR ratio {m['snr_ratio_pct']:.0f}%", j["result"].get("koma") or "")
    else:
        print(f"{engine}: FAILED - {j['error']}")
