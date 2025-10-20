#!/usr/bin/env python3
# hey_analyze.py — parse `hey` outputs (multiple runs), write CSV and charts
# Usage:
#   python hey_analyze.py -i hey_results.txt -o out_dir
#
# Only stdlib + (optional) matplotlib. If matplotlib is missing, it will skip plots.

import re
import csv
import os
import argparse
from typing import List, Dict, Any

def parse_hey_runs(text: str) -> List[Dict[str, Any]]:
    """
    Parse concatenated hey outputs. Returns list of dicts per run.
    Robust to optional 'Error distribution' section.
    """
    sections = re.split(r"===== Testing q=", text)
    if len(sections) <= 1:
        raise ValueError("No runs found. Make sure the input contains lines like '===== Testing q=... ====='.")

    runs = []
    for sec in sections[1:]:
        # Match header: q and estimated c
        m = re.match(
            r"(\d+),\s*estimated c=(\d+).*?Summary:\s*(.*?)Status code distribution:\s*(.*?)(?:Error distribution:\s*(.*?))?\nEnd time:",
            sec, re.S
        )
        if not m:
            # Some hey prints may vary; try a slightly looser end anchor
            m = re.match(
                r"(\d+),\s*estimated c=(\d+).*?Summary:\s*(.*?)Status code distribution:\s*(.*?)(?:Error distribution:\s*(.*?))?\n",
                sec, re.S
            )
        if not m:
            # Skip unparseable chunk instead of failing the whole file
            continue

        q = int(m.group(1))
        c = int(m.group(2))
        summary = m.group(3)
        status = m.group(4)
        errors = m.group(5) or ""

        def ffind(pattern, s=summary, cast=float):
            mm = re.search(pattern, s)
            if not mm: return None
            try:
                return cast(mm.group(1))
            except Exception:
                return None

        avg = ffind(r"Average:\s+([\d.]+)\s+secs")
        rps = ffind(r"Requests/sec:\s+([\d.]+)")
        slowest = ffind(r"Slowest:\s+([\d.]+)\s+secs")
        fastest = ffind(r"Fastest:\s+([\d.]+)\s+secs")
        total_secs = ffind(r"Total:\s+([\d.]+)\s+secs")

        # latency percentiles search over whole run section
        p90 = ffind(r"90% in ([\d.]+) secs", s=sec)
        p95 = ffind(r"95% in ([\d.]+) secs", s=sec)
        p99 = ffind(r"99% in ([\d.]+) secs", s=sec)

        # status code counts
        def sc_find(code):
            mm = re.search(rf"\[{code}\]\s+(\d+)\s+responses", status)
            return int(mm.group(1)) if mm else 0

        sc200 = sc_find(200)
        sc502 = sc_find(502)
        sc504 = sc_find(504)
        sc5xx_other = 0
        for code in (500, 501, 503, 505, 506, 507, 508, 509):
            sc5xx_other += sc_find(code)

        # error distribution counts
        def err_count(pattern):
            mm = re.search(pattern, errors)
            return int(mm.group(1)) if mm else 0

        err_timeout = err_count(r"\[(\d+)\]\s+Get .*?: context deadline exceeded")
        err_toomany_fd_tcp = err_count(r"\[(\d+)\]\s+Get .*?: dial tcp .*?: socket: too many open files")
        err_toomany_fd_dns = err_count(r"\[(\d+)\]\s+.*lookup .*?: dial udp .*?: socket: too many open files")
        err_dns_no_host = err_count(r"\[(\d+)\]\s+.*lookup .*?: no such host")

        # Derived helpers (Little's Law suggestions)
        c_est_avg = None if (avg is None) else int(round(q * avg))
        c_est_p95 = None if (p95 is None) else int(round(q * p95))

        runs.append({
            "q_target": q,
            "c_used": c,
            "avg_latency_s": avg,
            "p90_s": p90,
            "p95_s": p95,
            "p99_s": p99,
            "slowest_s": slowest,
            "fastest_s": fastest,
            "total_secs": total_secs,
            "achieved_rps": rps,
            "http_200": sc200,
            "http_502": sc502,
            "http_504": sc504,
            "http_5xx_other": sc5xx_other,
            "err_timeout": err_timeout,
            "err_too_many_open_files_tcp": err_toomany_fd_tcp,
            "err_too_many_open_files_dns": err_toomany_fd_dns,
            "err_dns_no_host": err_dns_no_host,
            "c_est_avg": c_est_avg,
            "c_est_p95": c_est_p95,
        })

    # sort by q then c
    runs.sort(key=lambda r: (r["q_target"], r["c_used"]))
    return runs

def write_csv(rows: List[Dict[str, Any]], out_csv: str):
    if not rows:
        raise ValueError("No parsed runs to write.")
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def try_plot(rows: List[Dict[str, Any]], out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[warn] matplotlib not available; skip plotting. Error:", e)
        return

    # Prepare series
    q = [r["q_target"] for r in rows]
    rps = [r["achieved_rps"] for r in rows]
    avg = [r["avg_latency_s"] for r in rows]
    p90 = [r["p90_s"] for r in rows]
    p95 = [r["p95_s"] for r in rows]
    p99 = [r["p99_s"] for r in rows]
    http504 = [r["http_504"] for r in rows]
    http502 = [r["http_502"] for r in rows]
    etimeout = [r["err_timeout"] for r in rows]
    efd_tcp = [r["err_too_many_open_files_tcp"] for r in rows]
    efd_dns = [r["err_too_many_open_files_dns"] for r in rows]
    enohost = [r["err_dns_no_host"] for r in rows]

    # Plot 1: Target vs Achieved RPS
    plt.figure()
    plt.plot(q, rps, marker="o", label="achieved_rps")
    plt.plot(q, q, linestyle="--", label="target_q (reference)")
    plt.xlabel("Target RPS (q)")
    plt.ylabel("Requests/sec")
    plt.title("Target vs Achieved RPS")
    plt.legend()
    p1 = os.path.join(out_dir, "hey_rps_vs_target.png")
    plt.savefig(p1, bbox_inches="tight")
    plt.close()

    # Plot 2: Latency percentiles vs q
    plt.figure()
    plt.plot(q, avg, marker="o", label="avg")
    plt.plot(q, p90, marker="o", label="p90")
    plt.plot(q, p95, marker="o", label="p95")
    plt.plot(q, p99, marker="o", label="p99")
    plt.xlabel("Target RPS (q)")
    plt.ylabel("Latency (s)")
    plt.title("Latency vs Target RPS")
    plt.legend()
    p2 = os.path.join(out_dir, "hey_latency_vs_q.png")
    plt.savefig(p2, bbox_inches="tight")
    plt.close()

    # Plot 3: Errors vs q
    plt.figure()
    plt.plot(q, http504, marker="o", label="HTTP 504")
    plt.plot(q, http502, marker="o", label="HTTP 502")
    plt.plot(q, etimeout, marker="o", label="Timeout")
    plt.plot(q, efd_tcp, marker="o", label="Too many open files (tcp)")
    plt.plot(q, efd_dns, marker="o", label="Too many open files (dns)")
    plt.plot(q, enohost, marker="o", label="DNS no such host")
    plt.xlabel("Target RPS (q)")
    plt.ylabel("Count")
    plt.title("Errors vs Target RPS")
    plt.legend()
    p3 = os.path.join(out_dir, "hey_errors_vs_q.png")
    plt.savefig(p3, bbox_inches="tight")
    plt.close()

    print("[ok] plots saved to:", p1, p2, p3)

def main():
    ap = argparse.ArgumentParser(description="Parse hey outputs into CSV and plots.")
    ap.add_argument("-i", "--input", required=True, help="Input text file containing concatenated hey outputs")
    ap.add_argument("-o", "--outdir", default=".", help="Output directory (default: current dir)")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    rows = parse_hey_runs(text)
    if not rows:
        raise SystemExit("No runs parsed. Check input content.")

    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, "hey_summary.csv")
    write_csv(rows, out_csv)
    print("[ok] CSV written:", out_csv)

    # Print quick topline (capacity hint)
    # Find q where achieved_rps stops increasing substantially (simple knee heuristic)
    knee_q = None
    prev = None
    for r in rows:
        if prev is not None:
            gain = (r["achieved_rps"] or 0) - (prev["achieved_rps"] or 0)
            if gain < 0.1 * (r["q_target"] - prev["q_target"]):  # crude knee rule
                knee_q = r["q_target"]
                break
        prev = r
    if knee_q:
        print(f"[hint] Achieved RPS growth flattens around q≈{knee_q}. Consider tuning concurrency or backend.")

    try_plot(rows, args.outdir)

    # Also print a tiny table summary to console
    print("\nq  c_used  achieved_rps  avg(s)  p95(s)  504  timeout  too_many_fd(dns)")
    for r in rows:
        print(f"{r['q_target']:>3} {r['c_used']:>6} {r['achieved_rps'] or 0:>13.1f} {r['avg_latency_s'] or 0:>6.3f} "
              f"{(r['p95_s'] or 0):>6.3f} {r['http_504']:>4} {r['err_timeout']:>8} {r['err_too_many_open_files_dns']:>16}")
    print("\n[tip] Suggested concurrency (Little's Law): c_est_avg & c_est_p95 are in CSV columns.")

if __name__ == "__main__":
    main()
