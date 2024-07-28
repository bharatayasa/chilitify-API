# chilitify-API
# FastAPI Image Prediction API

This repository contains a FastAPI application that provides an endpoint for image classification using a pre-trained TensorFlow model. The application uses the Keras library for model handling and image preprocessing.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/bharatayasa/chilitify-API.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd <project-directory>
    ```

3. **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

   If you don't have a `requirements.txt` file, you can manually install the dependencies with:
    ```bash
    pip install fastapi uvicorn tensorflow numpy pillow
    ```

## Model

Ensure that you have a pre-trained model saved as `model.h5` in the `./model/` directory. The model is loaded using TensorFlow/Keras.

## Running the Application

To start the FastAPI application, use:

```bash
uvicorn main:app --reload
