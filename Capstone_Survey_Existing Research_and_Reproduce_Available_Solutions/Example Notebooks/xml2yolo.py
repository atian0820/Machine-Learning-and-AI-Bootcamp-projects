import os
import re
import shutil
import random
import xml.etree.ElementTree as ET
from PIL import Image
from pathlib import Path

# Extended full class list based on your labels and text normalization
CLASSES = [
    'battery', 'buzzer', 'clock', 'diode', 'emi_filter', 'fuse', 'ic',
    'jumper', 'potentiometer', 'transformer', 'button', 'capacitor',
    'connector', 'display', 'ferrite_bead', 'heatsink', 'inductor',
    'led', 'resistor', 'transistor',
    'pads', 'pins', 'resistor_network', 'electrolytic_capacitor', 'text'
]

label_map = {name: i for i, name in enumerate(CLASSES)}

def clean_label(raw_label: str):
    # Lowercase & strip quotes, tidy spaces
    raw_label = raw_label.lower().replace('"', '').strip()

    # Regex to capture phrases like 'resistor jumper' or 'component text'
    match = re.match(r'([a-z_]+(?:\s[a-z_]+)?)', raw_label)
    if not match:
        return None

    main_token = match.group(1).replace(' ', '_')  # normalize with underscores

    # Special aliases to your CLASSES
    alias_map = {
        'resistor_network': 'resistor_network',
        'electrolytic_capacitor': 'electrolytic_capacitor',
        'emi_filter': 'emi_filter',
        'test_point': 'pads',
        'component_text': 'text',
        'pads': 'pads',
        'pins': 'pins',
        'text': 'text'
    }

    # First, exact alias mapping
    if main_token in alias_map:
        return alias_map[main_token]

    # Then, match against your class list
    if main_token in label_map:
        return main_token

    # Optional: partial fallback match
    for key in label_map.keys():
        if key in main_token:
            return key

    # Unknown label
    return None


def convert_xml_to_yolo(xml_path, image_path, label_path):
    img = Image.open(image_path)
    width, height = img.size

    tree = ET.parse(xml_path)
    root = tree.getroot()

    with open(label_path, 'w') as f_out:
        for obj in root.findall('object'):
            raw_label = obj.find('name').text
            cleaned_label = clean_label(raw_label)

            if cleaned_label is None:
                print(f"⚠️ Skipping unknown label: '{raw_label}' in {xml_path.name}")
                continue

            class_id = label_map[cleaned_label]

            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))

            x_center = ((xmin + xmax) / 2.0) / width
            y_center = ((ymin + ymax) / 2.0) / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

def create_dataset_structure(base_dir, split_ratio=0.8):
    base_dir = Path(base_dir)
    image_label_pairs = []

    # Collect xml+image pairs (.jpg or .png)
    for xml_file in base_dir.rglob("*.xml"):
        base_name = xml_file.stem
        # Try .jpg first, fallback to .png
        image_file = xml_file.with_suffix(".jpg")
        if not image_file.exists():
            image_file = xml_file.with_suffix(".png")
        if image_file.exists():
            image_label_pairs.append((image_file, xml_file))
        else:
            print(f"⚠️ Image file not found for {xml_file}")

    # Shuffle & split
    random.shuffle(image_label_pairs)
    split_index = int(len(image_label_pairs) * split_ratio)
    train_pairs = image_label_pairs[:split_index]
    val_pairs = image_label_pairs[split_index:]

    # Create folders
    for split in ['train', 'val']:
        (base_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (base_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Process conversion and copying
    for split, pairs in [('train', train_pairs), ('val', val_pairs)]:
        for image_path, xml_path in pairs:
            label_path = base_dir / split / 'labels' / (image_path.stem + '.txt')
            output_image_path = base_dir / split / 'images' / image_path.name

            try:
                convert_xml_to_yolo(xml_path, image_path, label_path)
                shutil.copy(image_path, output_image_path)
                print(f"[{split}] Processed: {image_path.name}")
            except Exception as e:
                print(f"[{split}] Failed: {image_path.name} — {e}")

def write_data_yaml(base_dir):
    yaml_path = Path(base_dir) / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names: {CLASSES}\n")

# === Run Conversion ===
if __name__ == "__main__":
    base_dir = "/Users/alan/Desktop/MLCapstone/pcb_wacv_2019"  # Change as needed
    create_dataset_structure(base_dir)
    write_data_yaml(base_dir)
