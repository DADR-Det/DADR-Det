from ultralytics import YOLO


def main():
    # Load trained weights
    model = YOLO(r"best.pt")

    # Validate on a custom dataset
    metrics = model.val(
        data=r"DIOR-R-filtered\data.yaml",
        batch=16,      # Batch size
        imgsz=640,     # Input image size
        rect=False,
        workers=8,
        device=0,      # GPU device
    )

    print(metrics.box.map)  # mAP50-95


if __name__ == "__main__":
    main()
