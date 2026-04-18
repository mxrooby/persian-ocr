# Persian Alphabet Character Recognition

A system that recognizes Persian alphabet characters from images and displays their Latin equivalents using a custom CNN model.

## Tech Stack

| Category | Tool |
|---|---|
| Model | Custom CNN (PyTorch) |
| Image Processing | OpenCV |
| User Interface | Gradio |
| Database | SQLite |
| Language | Python 3.11+ |

## Setup

```bash
git clone https://github.com/mxrooby/persian-ocr.git
cd persian-ocr
python -m venv venv
venv\Scripts\activate       # MacOS: source venv/bin/activate
pip install -r requirements.txt
```

## Train the Model

```bash
python train.py
```

* Trains the CNN using the dataset
* Automatically saves:
  * `model/persian_cnn.pth` (best model)
  * `model/label_map.json` (label mapping)

## Evaluate the Model

```bash
python test.py
```

* Loads the trained model
* Evaluates performance on the test dataset
* Outputs:
  * Accuracy per Persian character
  * Overall accuracy
  * Status based on threshold (e.g., Meets Threshold / Below Threshold)

## Run the App

```bash
python app.py
```

Then open in your browser:

```
http://127.0.0.1:7860
```

## How to Use

1. Upload an image of a Persian character
2. Click **Recognize**
3. The system displays:
   * Detected character
   * Character name
   * Latin equivalent
4. If the result is incorrect:
   * Click **Recognize again**
   * Up to **3 attempts** (uses top-3 predictions)

## Project Structure

```
persian-ocr/
├── app.py              # Gradio UI
├── train.py            # Model training script
├── test.py             # Model evaluation script
├── preprocessing.py    # Image preprocessing (OpenCV)
├── database.py         # SQLite database functions
├── dataset/            # Dataset
│   ├── train/
│   ├── test/
│   └── label_map.json
└── model/              # Saved model + label map
    ├── persian_cnn.pth
    └── label_map.json
```

## Notes

* Ensure the dataset is properly structured before training
* `train.py` will automatically save the best-performing model
* `test.py` requires the trained model and label map inside the `model/` folder
* For best results, use clear, centered images of characters