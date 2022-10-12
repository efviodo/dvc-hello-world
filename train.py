import os.path

from model import ConvNN


def main():
    # 1. Train model
    model = ConvNN(train_data_path="data/train.csv", val_data_path="data/test.csv", model_dir="model")
    model.train()
    model.save()

    # 2. Test model with a few images
    files = ["4.jpeg", "7.jpeg"]
    for f in files:
        prediction = model.predict(image_path=os.path.join("data", f))
        print(f"Image of a {os.path.splitext(f)[0]} amd model predicts: {prediction}")


if __name__ == "__main__":
    main()
