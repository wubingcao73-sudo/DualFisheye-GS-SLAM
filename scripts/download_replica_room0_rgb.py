#!/usr/bin/env python3
"""Download only the official Replica room0 RGB frames needed by demo.py."""

import argparse
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from remotezip import RemoteZip
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


URL = "https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"


def build_session():
    retry = Retry(
        total=10,
        connect=10,
        read=10,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("HEAD", "GET"),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/Replica/room0/colors")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Download only the first N sorted frames (for a quick smoke test).")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    with build_session() as session:
        with RemoteZip(URL, session=session) as archive:
            members = sorted(
                info.filename for info in archive.infolist()
                if info.filename.startswith("Replica/room0/results/frame")
                and info.filename.endswith(".jpg")
            )

    if args.limit is not None:
        members = members[:args.limit]
    if not members:
        raise RuntimeError("No room0 RGB frames were found in the remote archive.")

    worker_count = min(args.workers, len(members))
    groups = [members[index::worker_count] for index in range(worker_count)]
    lock = threading.Lock()
    completed = 0

    def download_group(group):
        nonlocal completed
        with build_session() as session:
            with RemoteZip(URL, session=session) as archive:
                for member in group:
                    target = output / Path(member).name
                    info = archive.getinfo(member)
                    if target.is_file() and target.stat().st_size == info.file_size:
                        with lock:
                            completed += 1
                        continue

                    temporary = target.with_suffix(target.suffix + ".part")
                    with archive.open(info) as source, temporary.open("wb") as destination:
                        shutil.copyfileobj(source, destination)

                    if temporary.stat().st_size != info.file_size:
                        raise RuntimeError(f"Size mismatch for {member}")
                    os.replace(temporary, target)

                    with lock:
                        completed += 1
                        if completed % 100 == 0 or completed == len(members):
                            print(f"Downloaded {completed}/{len(members)} RGB frames.", flush=True)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        list(executor.map(download_group, groups))

    print(f"Replica room0 RGB data is ready at {output.resolve()}")


if __name__ == "__main__":
    main()
