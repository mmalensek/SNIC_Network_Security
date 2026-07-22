#!/usr/bin/env python3
"""
Delete .json files under json_log/ (recursively) whose filename contains
a date older than N days. Handles names like:
  report_20240115.json
  report_20240115_153000.json
  report_20240115_153000_2.json
"""

import re
from pathlib import Path
from datetime import datetime, timedelta

ROOT_DIR = Path("json_log")
MAX_AGE_DAYS = 7

# Matches an 8-digit date (YYYYMMDD) anywhere in the filename.
# Adjust this pattern if your DATE_STAMP format is different
# (e.g. YYYY-MM-DD -> r"(\d{4}-\d{2}-\d{2})" and change the strptime format below).
DATE_PATTERN = re.compile(r"(\d{8})")
DATE_FORMAT = "%Y%m%d"


def extract_date(filename: str):
    match = DATE_PATTERN.search(filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), DATE_FORMAT)
    except ValueError:
        return None


def main():
    if not ROOT_DIR.is_dir():
        print(f"Folder not found: {ROOT_DIR.resolve()}")
        return

    cutoff = datetime.now() - timedelta(days=MAX_AGE_DAYS)
    deleted, skipped = 0, 0

    for path in ROOT_DIR.rglob("*.json"):
        file_date = extract_date(path.name)
        if file_date is None:
            print(f"[SKIP] No date found in name: {path}")
            skipped += 1
            continue

        if file_date < cutoff:
            try:
                path.unlink()
                print(f"[DELETED] {path} (date: {file_date.date()})")
                deleted += 1
            except OSError as e:
                print(f"[ERROR] Could not delete {path}: {e}")
        else:
            print(f"[KEEP] {path} (date: {file_date.date()})")

    print(f"\nDone. Deleted {deleted} file(s), skipped {skipped} unparseable file(s).")


if __name__ == "__main__":
    main()