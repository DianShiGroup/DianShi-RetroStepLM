"""
Download the runtime data files into `datasets/`.

Sources:

| File under datasets/                          | Source                                               |
|-----------------------------------------------|------------------------------------------------------|
| model_retro.pt                                | DESP desp_data.zip, figshare 25956076,               |
| idx2template_retro.json                       |   doi:10.6084/m9.figshare.25956076, CC BY 4.0        |
| uspto_190_targets.txt                         |   https://github.com/coleygroup/desp                 |
| canon_building_block_mol2idx_no_isotope.json  |                                                      |
| train_routes/routes_{train,val}.pkl           | LLM-Syn-Planner dataset.zip,                         |
|                                               |   https://github.com/zoom-wang112358/LLM-Syn-Planner |
| chembl_1000.pkl                               | PDVN Google Drive,                                   |
|                                               |   https://github.com/DiXue98/PDVN                    |
| scscore/model.ckpt-10654.as_numpy.json.gz     | https://github.com/connorcoley/scscore, MIT          |
| processed/building_blocks.parquet             | derived from the DESP dict, below                    |

Cite the upstream work you use.

Usage:
    python -m scripts.prepare_datasets
    python -m scripts.prepare_datasets keep_archives=True   # keep the downloaded archives
"""

import hashlib
import json
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

import chz

logger = logging.getLogger(__name__)

DATASETS = Path("datasets")
DOWNLOADS = DATASETS / "_downloads"

DESP_ZIP = "https://ndownloader.figshare.com/files/46831312"  # desp_data.zip, 9.5 GiB
LLM_SYN_PLANNER_ZIP = (
    "https://www.dropbox.com/scl/fi/dmmypid2ooohp3freiox8/dataset.zip"
    "?rlkey=fmrhvds6fmxck2cp8h94albpc&dl=1"  # 326 MiB
)
PDVN_CHEMBL = "https://drive.google.com/uc?export=download&id=1gWN0EVyT4dLl6u-jGPxOF7-dUhiDKJ64"
SCSCORE = (
    "https://raw.githubusercontent.com/connorcoley/scscore/master/"
    "models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz"
)

# zip member -> path under datasets/. Only what the released code loads; desp_data.zip
# also holds the forward, building-block, value and synthetic-distance models.
DESP_MEMBERS = {
    "model_retro.pt": "model_retro.pt",
    "idx2template_retro.json": "idx2template_retro.json",
    "uspto_190_targets.txt": "uspto_190_targets.txt",
    "canon_building_block_mol2idx_no_isotope.json": "canon_building_block_mol2idx_no_isotope.json",
}
ROUTE_MEMBERS = {
    "dataset/routes_train.pkl": "train_routes/routes_train.pkl",
    "dataset/routes_val.pkl": "train_routes/routes_val.pkl",
}

# verified byte-for-byte against the upstream releases
EXPECTED_MD5 = {
    "train_routes/routes_train.pkl": "4f95bb2d37e1e31a4c17befb2cb8c34f",
    "train_routes/routes_val.pkl": "6c647da0cd4c93e976afb24b7e3fcb86",
    "chembl_1000.pkl": "ce0d7957076127b64175b12b78269432",
}
EXPECTED_SIZE = {
    "model_retro.pt": 6759015273,
    "idx2template_retro.json": 64235919,
    "uspto_190_targets.txt": 18858,
    "canon_building_block_mol2idx_no_isotope.json": 1186132777,
    "scscore/model.ckpt-10654.as_numpy.json.gz": 6178760,
}


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"  have {dest.name}")
        return dest
    logger.info(f"  downloading {dest.name} from {url.split('?')[0]}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.rename(dest)
    return dest


def unzip(archive, members):
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for member, rel in members.items():
            dest = DATASETS / rel
            if dest.exists() and dest.stat().st_size > 0:
                logger.info(f"  have {rel}")
                continue
            if member not in names:
                raise SystemExit(
                    f"{member} not in {archive.name}; upstream layout changed.\n"
                    f"members: {sorted(names)}"
                )
            logger.info(f"  extracting {member} -> {rel}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out, length=1 << 20)


def build_parquet():
    """datasets/processed/building_blocks.parquet from DESP's SMILES -> idx dict."""
    import pandas as pd

    src = DATASETS / "canon_building_block_mol2idx_no_isotope.json"
    dest = DATASETS / "processed" / "building_blocks.parquet"
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"  have {dest.relative_to(DATASETS)}")
        return
    logger.info(f"  reading {src.name}")
    with open(src) as f:
        mol2idx = json.load(f)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"smiles": list(mol2idx), "idx": list(mol2idx.values())}).to_parquet(
        dest, index=False
    )
    logger.info(f"  wrote {len(mol2idx)} building blocks -> {dest.relative_to(DATASETS)}")


def verify():
    ok = True
    for rel, want in EXPECTED_MD5.items():
        path = DATASETS / rel
        if not path.exists():
            logger.error(f"  MISSING {rel}")
            ok = False
            continue
        h = hashlib.md5()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 22), b""):
                h.update(blk)
        if h.hexdigest() != want:
            logger.error(f"  md5 MISMATCH {rel}: {h.hexdigest()} != {want}")
            ok = False
        else:
            logger.info(f"  ok {rel}")
    for rel, want in EXPECTED_SIZE.items():
        path = DATASETS / rel
        if not path.exists():
            logger.error(f"  MISSING {rel}")
            ok = False
        elif path.stat().st_size != want:
            logger.error(f"  size MISMATCH {rel}: {path.stat().st_size} != {want}")
            ok = False
        else:
            logger.info(f"  ok {rel}")
    return ok


@chz.chz
class Config:
    keep_archives: bool = False
    """Keep the downloaded zips and the intermediate building-block dict."""


def main():
    args = chz.entrypoint(Config)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    DATASETS.mkdir(parents=True, exist_ok=True)

    logger.info("DESP (single-step model, templates, building blocks, USPTO-190):")
    unzip(download(DESP_ZIP, DOWNLOADS / "desp_data.zip"), DESP_MEMBERS)

    logger.info("LLM-Syn-Planner (USPTO train/val route database):")
    unzip(download(LLM_SYN_PLANNER_ZIP, DOWNLOADS / "dataset.zip"), ROUTE_MEMBERS)

    logger.info("PDVN (ChEMBL-1000 targets):")
    download(PDVN_CHEMBL, DATASETS / "chembl_1000.pkl")

    logger.info("SCScore (complexity model):")
    download(SCSCORE, DATASETS / "scscore" / "model.ckpt-10654.as_numpy.json.gz")

    logger.info("Deriving building_blocks.parquet:")
    build_parquet()

    logger.info("Verifying:")
    ok = verify()

    if not args.keep_archives:
        shutil.rmtree(DOWNLOADS, ignore_errors=True)
        (DATASETS / "canon_building_block_mol2idx_no_isotope.json").unlink(missing_ok=True)
        logger.info("Removed archives (pass keep_archives=True to keep them).")

    if not ok:
        raise SystemExit(1)
    logger.info(
        "\nDone. Next, with the tool server running, build the training parquet:\n"
        "  python -m tools.tool_server base_port=8100 num_instances=16\n"
        "  python -m slime_retro.prepare_data depth_min=5 depth_max=30\n"
    )


if __name__ == "__main__":
    main()
