"""Download the official PPG-DaLiA dataset (15 subjects, Reiss et al., Sensors 2019).

Source resolution (all official, no proxy/cache URL substitution):
  1. Original dataset page https://ubicomp.eti.uni-siegen.de/home/datasets/ic18w2/
     -> HTTP 404 (page retired; server ubi29.informatik.uni-siegen.de).
  2. Relocated official page https://ubi29.informatik.uni-siegen.de/usi/data_ppgdalia.html
     links the university's own Sciebo share
     https://uni-siegen.sciebo.de/s/pfHzlTepXkiJ4jP  ("PPG-DaLiA (Sensors'19)
     dataset (2.7 GB)"), a single file PPG_FieldStudy.zip, 2,864,337,783 bytes,
     served by WebDAV at https://uni-siegen.sciebo.de/public.php/dav/files/pfHzlTepXkiJ4jP
  3. UCI mirror https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip
     (data.zip, 2.7 GB) reachable but ~11 KB/s, no Range support -> fallback only.

Transfer notes (measured on this machine, 2026-11-15):
  * Direct HTTPS to Sciebo is throttled to ~13-16 KB/s per connection;
    parallel Range chunks scale linearly (8 -> ~110 KB/s, 24 -> ~300 KB/s).
  * The machine's configured system proxy (127.0.0.1:7897, Windows Internet
    Settings) reaches the SAME official URL at ~37 MB/s with curl, so the
    archive is fetched as one whole-file curl transfer through the system
    proxy (direct route as fallback).  Python requests through the same
    proxy stalled on response bodies, hence curl.  Later probes showed the
    server answering Range requests with the full body (206, bytes
    0-2864337782/2864337783), so chunked resume is not relied upon; each
    retry restarts the transfer (cheap at proxy speed).
  * Integrity: exact byte-size check, sha256 recorded, zipfile.testzip CRC
    check of every member, 15 per-subject pickles verified, one pickle
    inspected (keys/shapes/dtypes).

Outputs (all under out/_novelty/):
  dalia/PPG_FieldStudy.zip          the archive
  dalia/data/...                    unpacked per-subject pickles
  dalia_acquisition.json            url, sha256, bytes, pickle keys/shapes
  (stdout is redirected to dalia_download.log by the caller)
"""
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "dalia")
DATA = os.path.join(DEST, "data")
ZIP_PATH = os.path.join(DEST, "PPG_FieldStudy.zip")
STATUS_PATH = os.path.join(HERE, "dalia_acquisition.json")

ORIGINAL_PAGE = "https://ubicomp.eti.uni-siegen.de/home/datasets/ic18w2/"
OFFICIAL_PAGE = "https://ubi29.informatik.uni-siegen.de/usi/data_ppgdalia.html"
SHARE_URL = "https://uni-siegen.sciebo.de/s/pfHzlTepXkiJ4jP"
DAV_URL = "https://uni-siegen.sciebo.de/public.php/dav/files/pfHzlTepXkiJ4jP"
UCI_URL = "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
EXPECTED_BYTES = 2864337783          # Content-Range total, probed 2026-11-15
SYSTEM_PROXY = "http://127.0.0.1:7897"   # Windows Internet Settings proxy
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
ATTEMPTS = 6


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def curl_fetch(url, out_path, via_proxy, max_time=2400):
    """Whole-file curl transfer; abort if <50 KB/s for 60 s (stall guard)."""
    cmd = ["curl", "-sS", "-L", "--fail", "--max-time", str(max_time),
           "--speed-time", "60", "--speed-limit", "51200",
           "--retry", "2", "--retry-delay", "5",
           "-A", UA, "-o", out_path, "-w",
           "http=%{http_code} bytes=%{size_download} speed=%{speed_download}B/s",
           url]
    if via_proxy:
        cmd[1:1] = ["-x", SYSTEM_PROXY]
    log(f"curl {'via system proxy' if via_proxy else 'direct'}: {url}")
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=max_time + 120)
    log(f"  curl exit={p.returncode} {p.stdout.strip()} {p.stderr.strip()[:200]}")
    return p.returncode == 0


def inspect_pickle(pkl_path):
    import numpy as np
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")

    def describe(v):
        try:
            a = np.asarray(v)
            if a.dtype == object and a.shape == ():
                return {"type": "dict", "keys": sorted(v.keys())}
            return {"shape": list(a.shape), "dtype": str(a.dtype)}
        except Exception:
            return {"type": str(type(v))}

    info = {k: describe(v) for k, v in d.items()}
    if isinstance(d.get("signal"), dict):
        for part, sub in d["signal"].items():
            if isinstance(sub, dict):
                info[f"signal.{part}"] = {k: describe(v) for k, v in sub.items()}
    return info


def main():
    t0 = time.time()
    log(f"start; official source {DAV_URL}")
    os.makedirs(DATA, exist_ok=True)
    status = {
        "dataset": "PPG-DaLiA (Reiss et al., Sensors 2019, 15 subjects)",
        "original_page": ORIGINAL_PAGE,
        "original_page_status": "HTTP 404 (retired); official page relocated",
        "official_page": OFFICIAL_PAGE,
        "share_url": SHARE_URL,
        "url": DAV_URL,
        "uci_mirror_fallback": UCI_URL,
        "uci_mirror_note": "reachable but ~11 KB/s, no Range support -> not used",
        "filename": "PPG_FieldStudy.zip",
        "transfer": "whole-file curl via the machine's system proxy "
                    f"({SYSTEM_PROXY}); direct route fallback; same official "
                    "URL on both routes",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ok": False,
    }

    def bail(reason):
        status["ok"] = False
        status["reason"] = reason
        status["elapsed_s"] = round(time.time() - t0, 1)
        with open(STATUS_PATH, "w") as f:
            json.dump(status, f, indent=1)
        log(f"DOWNLOAD FAILED: {reason}")
        sys.exit(1)

    if os.path.exists(ZIP_PATH) and os.path.getsize(ZIP_PATH) == EXPECTED_BYTES:
        log("archive already present with expected size; skipping transfer")
    else:
        got = False
        for attempt in range(ATTEMPTS):
            via_proxy = (attempt % 2 == 0)     # proxy first (fast), direct fallback
            tmp = ZIP_PATH + ".part"
            if os.path.exists(tmp):
                os.remove(tmp)                 # server ignores Range: restart
            ok = curl_fetch(DAV_URL, tmp, via_proxy)
            if ok and os.path.exists(tmp) and os.path.getsize(tmp) == EXPECTED_BYTES:
                os.replace(tmp, ZIP_PATH)
                got = True
                break
            size = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            log(f"  attempt {attempt+1}/{ATTEMPTS} incomplete "
                f"({size}/{EXPECTED_BYTES} bytes); retrying")
            time.sleep(5 * (attempt + 1))
        if not got:
            bail(f"archive transfer failed after {ATTEMPTS} attempts")

    got = os.path.getsize(ZIP_PATH)
    if got != EXPECTED_BYTES:
        bail(f"archive size {got} != {EXPECTED_BYTES}")
    status["bytes"] = got
    log("sha256 ...")
    digest = sha256(ZIP_PATH)
    status["sha256"] = digest
    log(f"sha256 = {digest}")

    log("CRC-verifying every zip member (testzip) ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        bad = z.testzip()
        if bad is not None:
            bail(f"zip CRC failure in {bad}")
        names = z.namelist()
        log(f"zip OK: {len(names)} members; unpacking ...")
        z.extractall(DATA)
    status["zip_members"] = len(names)

    pkls = []
    for root, _, files in os.walk(DATA):
        for fn in files:
            if fn.lower().endswith(".pkl"):
                pkls.append(os.path.join(root, fn))
    pkls.sort()
    log(f"found {len(pkls)} pickle files")
    if len(pkls) != 15:
        bail(f"expected 15 subject pickles, found {len(pkls)}: {pkls}")
    status["pickles"] = [os.path.relpath(p, DATA).replace("\\", "/") for p in pkls]
    status["pickle_inspection"] = {
        os.path.basename(pkls[0]): inspect_pickle(pkls[0])}
    log(f"inspection of {os.path.basename(pkls[0])}: "
        f"{json.dumps(status['pickle_inspection'])}")

    status["ok"] = True
    status["elapsed_s"] = round(time.time() - t0, 1)
    with open(STATUS_PATH, "w") as f:
        json.dump(status, f, indent=1)
    log(f"DONE in {status['elapsed_s']}s -> {STATUS_PATH}")


if __name__ == "__main__":
    main()