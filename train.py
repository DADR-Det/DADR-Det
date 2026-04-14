from ultralytics import YOLO
import os


def main():
    # 1. Define a list of model configuration file paths
    model_yaml_paths = [
        r"DADR-Det yaml",
        # r"ultralytics/cfg/models/yolo11-obb.yaml",
        # Add more model configuration paths here
    ]

    # 2. Train each model configuration in sequence
    for model_yaml_path in model_yaml_paths:
        try:
            # Check whether the configuration file exists
            if not os.path.exists(model_yaml_path):
                print(f"Warning: configuration file not found: {model_yaml_path}")
                continue

            # Extract the file name without extension as the experiment name
            model_name = os.path.splitext(os.path.basename(model_yaml_path))[0]

            print("\n" + "=" * 60)
            print(f"Start training model: {model_name}")
            print(f"Configuration path: {model_yaml_path}")
            print("=" * 60)

            # 3. Load model
            model = YOLO(model_yaml_path, task="obb")

            # 4. Train model with the dynamically generated run name
            results = model.train(
                data=r"DIOR-R-filtered\data.yaml",
                epochs=200,
                imgsz=640,
                batch=16,
                workers=8,
                optimizer="SGD",
                patience=100,
                # resume=True,
                project="runs/11-obb",
                name=model_name,
                device=0,
            )

            print(f"Training completed for model: {model_name}")

        except Exception as e:
            print(f"Training failed for model {model_yaml_path}: {str(e)}")
            continue

    print("\n" + "=" * 60)
    print("All model training jobs have finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()
