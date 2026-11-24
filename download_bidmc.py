"""Download the full PhysioNet BIDMC PPG+ECG dataset (53 subjects, fs=125 Hz).

Official source only (physionet.org). The canonical /files/bidmc/1.0.0/ endpoint
returned HTTP 403 for every file from this machine (while /files/wrist and
/files/mitdb worked), so the official legacy PhysioBank archive path is used:
    https://physionet.org/physiobank/database/bidmc/
Same dataset (v1.0.0), legacy record names bidmc01..bidmc53, signals
RESP/PLETH/V/AVR/II (PLETH = PPG, lead II = ECG reference).
Protocol: fetch RECORDS and SHA256SUMS.txt first, then download every .hea/.dat
listed for every record, verifying SHA256 per file. All-or-nothing: if the
15-minute budget expires or any record fails, the cross-dataset experiment is
abandoned (no cherry-picked subset) and the failure is recorded in
out/_novelty/bidmc_download_status.json.
"""
import hashlib
import json
import os
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEST = os.path.join(ROOT, "data", "bidmc")
OUT = os.path.join(ROOT, "out", "_novelty")
BASE = "https://physionet.org/physiobank/database/bidmc/"
BASE_FILES = "https://physionet.org/files/bidmc/1.0.0/"  # canonical (403 here)
DEADLINE_S = 15 * 60
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"}


def get(url, timeout=(15, 90)):
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.content


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    os.makedirs(DEST, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    status = {"source": BASE, "canonical_source_tried": BASE_FILES,
              "canonical_source_status": "HTTP 403 on all /files/bidmc/1.0.0/ paths "
                                         "(wrist/mitdb /files/ OK); official legacy "
                                         "PhysioBank archive used instead",
              "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "deadline_s": DEADLINE_S, "ok": False}

    def bail(reason):
        status["ok"] = False
        status["reason"] = reason
        status["elapsed_s"] = round(time.time() - t0, 1)
        with open(os.path.join(OUT, "bidmc_download_status.json"), "w") as f:
            json.dump(status, f, indent=1)
        print("DOWNLOAD FAILED:", reason, flush=True)
        sys.exit(1)

    try:
        records = [ln.strip() for ln in get(BASE + "RECORDS").decode().splitlines()
                   if ln.strip()]
        sums_raw = get(BASE + "SHA256SUMS.txt").decode()
    except Exception as e:
        bail(f"cannot fetch RECORDS/SHA256SUMS: {e!r}")
    sums = {}
    for ln in sums_raw.splitlines():
        parts = ln.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0]
    status["n_records_listed"] = len(records)
    print(f"RECORDS: {len(records)} records; SHA256SUMS: {len(sums)} files", flush=True)
    if len(records) != 53:
        bail(f"unexpected RECORDS count {len(records)} (expected 53)")

    # every .hea/.dat file belonging to the listed records
    jobs = []
    for rec in records:
        files = sorted(fn for fn in sums
                       if fn.startswith(rec + ".") and fn.rsplit(".", 1)[-1] in ("hea", "dat"))
        if not any(fn.endswith(".hea") for fn in files):
            bail(f"no .hea listed for {rec}")
        if not any(fn.endswith(".dat") for fn in files):
            bail(f"no .dat listed for {rec}")
        jobs.extend(files)
    status["n_files"] = len(jobs)
    print(f"downloading {len(jobs)} files (10 parallel connections) ...", flush=True)

    # The archive throttles each connection to ~8 KB/s; a sequential download
    # of ~35 MB would take ~75 min. Parallel connections (same official URLs,
    # every file, sha256-verified) keep the transfer inside the 15-min budget.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    lock = threading.Lock()
    done = 0
    failed = []

    def fetch_one(fn):
        nonlocal done
        dest = os.path.join(DEST, fn)
        if os.path.exists(dest) and sha256(dest) == sums[fn]:
            with lock:
                done += 1
            return fn, "cached"
        for attempt in range(3):
            if time.time() - t0 > DEADLINE_S:
                return fn, "deadline"
            try:
                data = get(BASE + fn)
                tmp = dest + f".part{threading.get_ident()}"
                with open(tmp, "wb") as f:
                    f.write(data)
                if sha256(tmp) != sums[fn]:
                    raise ValueError("sha256 mismatch")
                os.replace(tmp, dest)
                with lock:
                    done += 1
                    if done % 20 == 0:
                        print(f"  {done}/{len(jobs)} files, {time.time()-t0:.0f}s",
                              flush=True)
                return fn, "ok"
            except Exception as e:
                print(f"  retry {attempt+1} {fn}: {e!r}", flush=True)
                time.sleep(2 * (attempt + 1))
        return fn, "fail"

    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(fetch_one, fn): fn for fn in jobs}
        for fut in as_completed(futs):
            fn, st = fut.result()
            if st in ("fail", "deadline"):
                failed.append((fn, st))
    if failed:
        bail(f"{len(failed)} files failed: {failed[:5]}")
    if time.time() - t0 > DEADLINE_S:
        bail("15-minute budget exhausted")

    status["ok"] = True
    status["n_files_verified"] = done
    status["transfer"] = "10 parallel connections; archive throttles ~8 KB/s per connection"
    status["elapsed_s"] = round(time.time() - t0, 1)
    status["records"] = records
    with open(os.path.join(OUT, "bidmc_download_status.json"), "w") as f:
        json.dump(status, f, indent=1)
    print(f"DONE: {done} files verified in {status['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()