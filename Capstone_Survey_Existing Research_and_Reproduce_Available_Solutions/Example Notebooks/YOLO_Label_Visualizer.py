import os
import cv2
import argparse
import matplotlib.pyplot as plt
import matplotlib

# Use TkAgg for script mode pop-up windows
matplotlib.use('TkAgg')

CLASSES = [
    'battery', 'buzzer', 'clock', 'diode', 'emi_filter', 'fuse', 'ic',
    'jumper', 'potentiometer', 'transformer', 'button', 'capacitor',
    'connector', 'display', 'ferrite_bead', 'heatsink', 'inductor',
    'led', 'resistor', 'transistor'
]

def visualize_yolo_labels(image_path: str, label_path: str):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    if not os.path.exists(label_path):
        print(f"❌ Label file not found: {label_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    print(f"📝 Found {len(lines)} annotations in {os.path.basename(label_path)}")

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id, x_center, y_center, width, height = map(float, parts)
        class_id = int(class_id)

        x_center *= img_w
        y_center *= img_h
        width *= img_w
        height *= img_h

        xmin = int(x_center - width / 2)
        ymin = int(y_center - height / 2)
        xmax = int(x_center + width / 2)
        ymax = int(y_center + height / 2)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        label = CLASSES[class_id] if class_id < len(CLASSES) else f"Unknown({class_id})"
        cv2.putText(image, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Annotations: {os.path.basename(image_path)}")
    plt.show(block=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize YOLO labels on an image")
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--label', type=str, required=True, help='Path to YOLO .txt label file')
    args = parser.parse_args()

    visualize_yolo_labels(args.image, args.label)
