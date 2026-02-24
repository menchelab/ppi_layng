"""
Step 1: Download GO ontology (go-basic.obo) and GAF (goa_human) to data/.

Do manually if 403, just two links from config.py and decompress + put in data/

"""
import gzip
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

from tqdm import tqdm

from config_tune import DATA_DIR, GO_OBO_PATH, GAF_PATH, GO_OBO_URL, GAF_URL, ensure_dirs

CHUNK = 1024 * 1024  # 1 MiB


def _download_with_progress(url, path, desc="Downloading"):
    resp = urlopen(url)
    total = int(resp.headers.get("Content-Length", 0)) or None
    with open(path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=desc) as pbar:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
    return path


def download_obo():
    path = GO_OBO_PATH
    if path.exists():
        print(f"Already exists: {path}")
        return path
    ensure_dirs()
    _download_with_progress(GO_OBO_URL, path, desc="go-basic.obo")
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Download failed or empty: {path}")
    print(f"Saved: {path} ({path.stat().st_size} bytes)")
    return path


def download_gaf():
    path = GAF_PATH
    if path.exists():
        print(f"Already exists: {path}")
        return path
    ensure_dirs()
    gz_path = path.with_suffix(path.suffix + ".gz")
    _download_with_progress(GAF_URL, gz_path, desc="goa_human.gaf.gz")
    if not gz_path.exists() or gz_path.stat().st_size == 0:
        raise RuntimeError(f"Download failed or empty: {gz_path}")
    with gzip.open(gz_path, "rb") as f_in:
        with open(path, "wb") as f_out:
            with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc="Decompressing GAF") as pbar:
                while True:
                    chunk = f_in.read(CHUNK)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    pbar.update(len(chunk))
    gz_path.unlink()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Decompress failed or empty: {path}")
    print(f"Saved: {path} ({path.stat().st_size} bytes)")
    return path


def main():
    ensure_dirs()
    download_obo()
    download_gaf()
    # Validate
    if not GO_OBO_PATH.exists():
        print("ERROR: OBO file missing after download.", file=sys.stderr)
        sys.exit(1)
    if not GAF_PATH.exists():
        print("ERROR: GAF file missing after download.", file=sys.stderr)
        sys.exit(1)
    print("Step 1 done. Data files ready.")


if __name__ == "__main__":
    main()
