"""
Parallel batch converter for Google Scanned Objects ZIPs → model.urdf

Usage:
    python extract_GSO.py <path_to_gso_zips>  [-o <output_root>] [-w 8]

Each ZIP is unpacked into <output_root>/<zip_stem>/...
and a model.urdf is created alongside model.sdf.
"""

import argparse, zipfile, multiprocessing as mp
from pathlib import Path
from functools import partial
from tqdm import tqdm

# ANSI colours
GREEN, YELLOW, RED, RESET = "\033[92m", "\033[93m", "\033[91m", "\033[0m"

URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="{name}">
  <link name="base">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="meshes/model.obj" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="meshes/model.obj" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
</robot>
"""


def patch_mtl(mtl_path: Path):
    lines = mtl_path.read_text().splitlines()
    new_lines = []
    for ln in lines:
        if ln.startswith("map_Kd "):
            tex = ln.split(maxsplit=1)[1].strip()
            if "/" not in tex:
                ln = f"map_Kd ../materials/textures/{tex}"
        new_lines.append(ln)
    mtl_path.write_text("\n".join(new_lines))


def process_zip(zip_path: Path, out_root: Path) -> str:
    """Unpack single zip & drop model.urdf; return coloured status string."""
    if zip_path.suffix.lower() != ".zip":
        return f"{YELLOW}[skip]{RESET} {zip_path.name} (not a zip)"

    out_dir = out_root / zip_path.stem
    try:
        if out_dir.exists():
            return f"{YELLOW}[skip]{RESET} {zip_path.name}"

        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)

        patch_mtl(out_dir / "meshes/model.mtl")

        urdf = out_dir / "model.urdf"
        if not urdf.exists():
            urdf.write_text(URDF_TEMPLATE.format(name=zip_path.stem))

        return f"{GREEN}[ok]{RESET}   {zip_path.name}"
    except Exception as e:
        return f"{RED}[err]{RESET}  {zip_path.name}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_root", type=Path, help="directory that contains *.zip models")
    ap.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=None,
        help="where to unpack (default: alongside each zip)",
    )
    ap.add_argument(
        "-w",
        "--workers",
        type=int,
        default=mp.cpu_count(),
        help="number of parallel workers (default = cpu_count)",
    )
    args = ap.parse_args()

    out_root = args.output_root or args.zip_root
    zips = list(args.zip_root.glob("*.zip"))
    if not zips:
        print("No *.zip found in", args.zip_root)
        return

    worker = partial(process_zip, out_root=out_root)
    with mp.Pool(processes=args.workers) as pool, tqdm(total=len(zips), desc="Converting") as bar:
        for msg in pool.imap_unordered(worker, zips):
            bar.write(msg)
            bar.update()


if __name__ == "__main__":
    main()
