# Persian Alphabet Character Recognition

A machine learning-based system that recognizes Persian alphabet characters from images and outputs their corresponding Latin/Romanized equivalents.

## Project Overview

This system accepts an image containing a Persian character, preprocesses it using OpenCV, and passes it through an EasyOCR-based CRNN model with CTC decoding to recognize the character. The recognized character and its Romanized equivalent are then displayed through a Gradio web interface. All recognition history is stored in a local SQLite database.

## Tech Stack

| Category | Tool |
|---|---|
| Technique | End-to-End Scene Text Recognition (STR) |
| Model Architecture | CRNN (CNN + LSTM) with CTC Loss |
| Implementation Library | EasyOCR |
| Programming Language | Python 3.11+ |
| Image Processing | OpenCV |
| User Interface | Gradio |
| Database | SQLite |
| Development IDE | Visual Studio Code |
| Version Control | Git |

## Project Structure
persian-ocr/
├── app.py              # Main application — UI and recognition logic
├── preprocessing.py    # Image preprocessing using OpenCV
├── database.py         # SQLite database setup and operations
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

## Setup Instructions

1. Clone the repository
git clone https://github.com/mxrooby/persian-ocr.git
cd persian-ocr

2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

3. Install dependencies
python -m pip install -r requirements.txt

4. Run the application
python app.py

5. Open your browser and go to
http://127.0.0.1:7860

## How to Use

1. Upload, capture, or paste an image of a Persian character
2. Crop the image tightly around the character if needed
3. Click **Recognize**
4. The system will display the recognized Persian character and its Romanized equivalent
5. All recognitions are saved to the history table automatically
