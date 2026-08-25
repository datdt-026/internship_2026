#!/usr/bin/env python3
"""Download MMCoVaR news images from URL column into data/raw/mmcovar/images."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "mmcovar" / "MMCoVaR_News_Dataset.csv"
IMG_DIR = ROOT / "data" / "raw" / "mmcovar" / "images"
MANIFEST_PATH = ROOT / "data" / "raw" / "mmcovar" / "image_download_manifest.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=str(CSV_PATH))
    p.add_argument("--out-dir", type=str, default=str(IMG_DIR))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout", type=int, default=20)
    p.add_argument("--limit", type=int, default=None, help="Optional cap for smoke tests")
    p.add_argument("--retries", type=int, default=2)
    return p.parse_args()


def guess_ext(url: str, content_type: str | None) -> str:
    url_l = url.lower().split("?")[0]
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        if url_l.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    if content_type:
        if "png" in content_type:
            return ".png"
        if "webp" in content_type:
            return ".webp"
        if "gif" in content_type:
            return ".gif"
    return ".jpg"


def download_one(
    news_id: str,
    url: str,
    out_dir: Path,
    timeout: int,
    retries: int,
) -> dict:
    # Skip if any existing file for this id
    existing = list(out_dir.glob(f"{news_id}.*"))
    if existing:
        return {
            "news_id": news_id,
            "url": url,
            "status": "exists",
            "path": str(existing[0].relative_to(ROOT)),
        }

    last_err = None
    for attempt in range(retries + 1):
        try:
            req = Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; USTH-ThesisBot/1.0)",
                    "Accept": "image/*,*/*;q=0.8",
                },
            )
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                ctype = resp.headers.get("Content-Type", "")
            if not data or len(data) < 100:
                raise ValueError(f"empty/too-small image ({len(data)} bytes)")
            ext = guess_ext(url, ctype)
            path = out_dir / f"{news_id}{ext}"
            path.write_bytes(data)
            return {
                "news_id": news_id,
                "url": url,
                "status": "ok",
                "path": str(path.relative_to(ROOT)),
                "bytes": len(data),
            }
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as e:
            last_err = str(e)
            time.sleep(0.4 * (attempt + 1))
    return {"news_id": news_id, "url": url, "status": "fail", "error": last_err}


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["news_id", "image"])
    df["news_id"] = df["news_id"].astype(str)
    df["image"] = df["image"].astype(str)
    if args.limit:
        df = df.head(args.limit)

    jobs = [(row.news_id, row.image) for row in df.itertuples(index=False)]
    print(f"[download] {len(jobs)} images -> {out_dir} (workers={args.workers})")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(download_one, nid, url, out_dir, args.timeout, args.retries): nid
            for nid, url in jobs
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="images"):
            results.append(fut.result())

    ok = sum(1 for r in results if r["status"] in {"ok", "exists"})
    fail = sum(1 for r in results if r["status"] == "fail")
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[download] ok/exists={ok} fail={fail}")
    print(f"[download] manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
