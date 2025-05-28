import os
from pathlib import Path
import yaml

# === CONFIG ===
yaml_path = '/Users/alan/Desktop/MLCapstone/pcb_wacv_2019/data.yaml'
dataset_root = Path(yaml_path).parent

# === Load YAML ===
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

print(f"Loaded YAML: {yaml_path}")
print(f"Classes ({data['nc']}): {data['names']}")

# === Check Paths ===
train_images = dataset_root / data['train']
val_images = dataset_root / data['val']

for split, img_dir in [('train', train_images), ('val', val_images)]:
    if not img_dir.exists():
        print(f"❌ ERROR: {split} image directory not found: {img_dir}")
    else:
        print(f"✅ Found {split} images directory: {img_dir}")

# === Check Labels ===
def check_labels(img_dir):
    label_dir = img_dir.parent / 'labels'
    if not label_dir.exists():
        print(f"❌ ERROR: labels directory missing: {label_dir}")
        return

    issues = 0
    for label_file in label_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"⚠️ Malformed line in {label_file.name} (line {line_num}): {line.strip()}")
                issues += 1
                continue

            try:
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
            except ValueError:
                print(f"⚠️ Non-numeric value in {label_file.name} (line {line_num}): {line.strip()}")
                issues += 1
                continue

            if not (0 <= class_id < data['nc']):
                print(f"⚠️ Class ID {class_id} out of range in {label_file.name} (line {line_num})")
                issues += 1

            for coord in coords:
                if not (0.0 <= coord <= 1.0):
                    print(f"⚠️ Bounding box value out of range (0-1) in {label_file.name} (line {line_num}): {line.strip()}")
                    issues += 1

    if issues == 0:
        print(f"✅ Labels in {label_dir} look good!")
    else:
        print(f"❗ Found {issues} issues in {label_dir}")

# Run checks
check_labels(train_images)
check_labels(val_images)

print("Dataset validation done.")
