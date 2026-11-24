"""Resilient PhysioNet fetcher v2 - stores files with original record names
inside per-dataset folders so wfdb can read them directly."""
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 research-fetch"}

WRIST_RECORDS = [
    "s1_high_resistance_bike", "s1_low_resistance_bike", "s1_walk",
    "s2_high_resistance_bike", "s2_low_resistance_bike", "s2_walk",
    "s3_high_resistance_bike", "s3_low_resistance_bike", "s3_run", "s3_walk",
    "s4_run", "s5_low_resistance_bike", "s5_run",
    "s6_low_resistance_bike", "s6_run", "s6_walk",
    "s8_run", "s8_walk", "s9_walk",
]
MITDB_RECORDS = ["100", "101", "103", "105", "106", "107", "108", "109", "111",
                 "112", "114", "116", "118", "119", "121", "122", "123",
                 "200", "202", "203", "205", "207", "208", "209", "210",
                 "212", "213", "214", "215", "217", "219", "220", "221",
                 "222", "223", "228", "230", "231", "232", "233", "234"]
BIDMC_RECORDS = [f"bidmc_{i:02d}" for i in range(1, 54)]


def is_html(path):
    try:
        with open(path, "rb") as f:
            head = f.read(512).lower()
        return b"<!doctype html" in head or b"<html" in head
    except Exception:
        return False


def expected_dat_size(hea_path):
    """Bytes a .dat must occupy, derived from the .hea header.

    Record line: '<name> <nsig> <fs> <nsamp>'. Each signal line starts with the
    file name then the storage format. WFDB byte costs:
      16  -> 2 bytes/sample        80 -> 1 byte/sample
      212 -> 1.5 bytes/sample (two 12-bit samples packed in 3 bytes)
      310/311 -> 1.5 (three 10-bit in 4 bytes) / 312 -> 1.5
    """
    try:
        with open(hea_path) as f:
            lines = [ln.split() for ln in f if ln.strip() and not ln.startswith("#")]
        if not lines or len(lines[0]) < 4:
            return None
        nsig, nsamp = int(lines[0][1]), int(lines[0][3])
        # signal lines follow the record line; a line is a signal line when it
        # has >=2 fields and its second field parses as a WFDB format code.
        cost = 0.0
        counted = 0
        for ln in lines[1:]:
            if len(ln) < 2:
                continue
            try:
                fmt = int(ln[1].split("x")[0].split(":")[0])
            except ValueError:
                continue
            bps = {16: 2.0, 61: 2.0, 80: 1.0, 160: 2.0,
                   212: 1.5, 310: 1.5, 311: 1.5, 312: 1.5}.get(fmt)
            if bps is None:
                return None
            cost += bps
            counted += 1
        if counted != nsig:
            return None
        return int(round(nsamp * cost))
    except Exception:
        return None


def ok_file(path, minbytes=10):
    if not (os.path.exists(path) and os.path.getsize(path) >= minbytes and not is_html(path)):
        return False
    if path.endswith(".dat"):
        exp = expected_dat_size(path[:-4] + ".hea")
        if exp is not None and os.path.getsize(path) != exp:
            return False
    return True


def fetch(url, dest, max_attempts=8, timeout=120):
    if ok_file(dest):
        return "cached", os.path.getsize(dest)
    delay = 2.0
    for _ in range(max_attempts):
        tmp = dest + ".part"
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                while True:
                    c = r.read(1 << 16)
                    if not c:
                        break
                    f.write(c)
            if ok_file(tmp):
                os.replace(tmp, dest)
                return "ok", os.path.getsize(dest)
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        time.sleep(delay)
        delay = min(delay * 1.7, 30)
    return "fail", 0


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    logf = open(os.path.join(DATA, "_download_log.txt"), "a")
    groups = {
        "wrist": (os.path.join(DATA, "wrist"),
                  [(rec, f"https://physionet.org/files/wrist/1.0.0/{rec}.{e}?download",
                    f"{rec}.{e}") for rec in WRIST_RECORDS for e in ("dat", "hea")]),
        "mitdb": (os.path.join(DATA, "mitdb"),
                  [(rec, f"https://physionet.org/files/mitdb/1.0.0/{rec}.{e}",
                    f"{rec}.{e}") for rec in MITDB_RECORDS for e in ("dat", "hea", "atr")]),
        "bidmc": (os.path.join(DATA, "bidmc"),
                  [(rec, f"https://physionet.org/files/bidmc/1.0.0/bidmc_csv/{rec}_{k}.csv?download",
                    f"{rec}_{k}.csv") for rec in BIDMC_RECORDS for k in ("Signals", "Numerics")]),
    }
    names = list(groups) if which == "all" else [which]
    for gname in names:
        folder, jobs = groups[gname]
        os.makedirs(folder, exist_ok=True)
        got = 0
        for label, url, fn in jobs:
            st, n = fetch(url, os.path.join(folder, fn))
            if st in ("ok", "cached"):
                got += 1
            logf.write(f"{gname}\t{label}\t{st}\t{n}\n")
            logf.flush()
            print(f"[{gname}] {label} {st} {n}", flush=True)
        print(f"[{gname}] DONE {got}/{len(jobs)}", flush=True)
    logf.close()


if __name__ == "__main__":
    main()
