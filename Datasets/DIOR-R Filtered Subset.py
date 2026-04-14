import shutil
from pathlib import Path


# Original DIOR-R class ids -> new class ids
CLASS_ID_MAP = {
    0: 0,    # airplane -> airplane
    18: 1,   # vehicle  -> vehicle
    13: 2,   # ship     -> ship
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def filter_and_remap_label(src_label: Path, dst_label: Path) -> bool:
    """
    Keep only airplane / vehicle / ship from the original 20-class labels,
    and remap them to new ids: airplane=0, vehicle=1, ship=2.

    Returns:
        True  -> at least one valid object remains
        False -> no valid object remains
    """
    if not src_label.exists():
        return False

    kept_lines = []

    with src_label.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            try:
                old_cls = int(parts[0])
            except ValueError:
                continue

            if old_cls in CLASS_ID_MAP:
                new_cls = CLASS_ID_MAP[old_cls]
                parts[0] = str(new_cls)
                kept_lines.append(" ".join(parts))

    if not kept_lines:
        return False

    with dst_label.open("w", encoding="utf-8") as f:
        f.write("\n".join(kept_lines) + "\n")

    return True


def process_split(src_root: Path, dst_root: Path, split: str):
    src_img_dir = src_root / "images" / split
    src_lbl_dir = src_root / "labels" / split

    dst_img_dir = dst_root / "images" / split
    dst_lbl_dir = dst_root / "labels" / split

    ensure_dir(dst_img_dir)
    ensure_dir(dst_lbl_dir)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    kept_images = 0
    removed_images = 0

    for img_path in src_img_dir.iterdir():
        if img_path.suffix.lower() not in image_exts:
            continue

        src_label = src_lbl_dir / f"{img_path.stem}.txt"
        dst_label = dst_lbl_dir / f"{img_path.stem}.txt"

        has_valid_objects = filter_and_remap_label(src_label, dst_label)

        if has_valid_objects:
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            kept_images += 1
        else:
            removed_images += 1
            if dst_label.exists():
                dst_label.unlink()

    print(f"[{split}] kept: {kept_images}, removed: {removed_images}")


def write_data_yaml(dst_root: Path):
    yaml_text = """path: .
train: images/train
val: images/val

names:
  0: airplane
  1: vehicle
  2: ship
"""
    with (dst_root / "data.yaml").open("w", encoding="utf-8") as f:
        f.write(yaml_text)


def main():
    # Change these paths to your actual dataset paths
    src_root = Path(r"DIOR-R")
    dst_root = Path(r"DIOR-R-filtered")

    ensure_dir(dst_root)

    for split in ["train", "val"]:
        process_split(src_root, dst_root, split)

    write_data_yaml(dst_root)
    print("Filtering finished.")


if __name__ == "__main__":
    main()
