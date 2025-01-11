# FastAPI ONNX Model Inference API

This repository contains a FastAPI application for performing inference using an ONNX model. The app uses a pre-trained `Roberta` model fine-tuned for sequence classification tasks, providing predictions on whether a given phrase is positive or not.

---

## Features
- Exposes an HTTP API to interact with an ONNX model.
- Tokenizes input text using `Hugging Face Transformers`.
- Supports POST requests for predictions.
- Lightweight and easy-to-deploy FastAPI app.

---

## Requirements

Ensure you have the following installed:

- Python 3.8+
- pip

### Python Dependencies:
Install the required Python libraries:
```bash
pip install fastapi uvicorn transformers torch numpy onnxruntime
```

---

## Directory Structure
```
fastapi-roberta
├── webapp                             # Contains the FastAPI application
│   ├── main.py                        # FastAPI application code
│   ├── roberta-sequence-classification-9-finetuned.onnx  # ONNX model file
├── .gitignore                         # Git ignore file
├── Dockerfile                         # Docker configuration
├── LICENSE                            # License for the project
├── README.md                          # Project documentation
├── requirements.txt                   # Dependencies
```

---

## Usage

### 1. Start the FastAPI Server
Run the following command to start the server:
```bash
uvicorn webapp.main:app --reload
```

By default, the server will run on `http://127.0.0.1:8000/`.

### 2. Test the API

#### Root Endpoint:
```bash
curl http://127.0.0.1:8000/
```
**Response:**
```html
<h1>A self-documenting API to interact with an ONNX model</h1>
```

#### Prediction Endpoint:
Make a POST request to the `/predict` endpoint with a JSON payload containing a phrase.

##### Example Request:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"phrase": "This is a great example!"}'
```

##### Example Response:
```json
{
  "positive": true
}
```

---

## Code Explanation

### Tokenization
The app uses the `RobertaTokenizer` from Hugging Face to tokenize input phrases into:
- `input_ids`
- `attention_mask`

### Inference
The ONNX model performs inference using the tokenized inputs:
- Converts PyTorch tensors to NumPy arrays using the `to_numpy` function.
- Runs the model using `onnxruntime.InferenceSession`.
- Returns whether the phrase is positive or not based on the model's output.

---

## Deployment

### Local Deployment
1. Clone the repository.
2. Install dependencies.
3. Start the server:
   ```bash
   uvicorn webapp.main:app --host 0.0.0.0 --port 8000
   ```

### Docker Deployment
1. Use the provided `Dockerfile`.

2. Build and run the Docker image:
   ```bash
   docker build -t fastapi-onnx-app .
   docker run -p 8000:8000 fastapi-onnx-app
   ```

---

## Notes
- The ONNX model file is located in the `webapp` directory.
- Modify the model path in `webapp/main.py` if needed:
  ```python
  session = onnxruntime.InferenceSession("path/to/your/onnx/model.onnx")
  ```
