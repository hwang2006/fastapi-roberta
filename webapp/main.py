from fastapi import FastAPI, Response
from pydantic import BaseModel

import torch
import numpy as np
from transformers import RobertaTokenizer
import onnxruntime


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
#session = onnxruntime.InferenceSession("roberta-sequence-classification-9.onnx")
session = onnxruntime.InferenceSession("../../roberta-sequence-classification-9-finetuned.onnx")


class Body(BaseModel):
    phrase: str


app = FastAPI()


@app.get('/')
def root():
    return Response("<h1>A self-documenting API to interact with an ONNX model</h1>")


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    return tensor.cpu().numpy()

@app.post('/predict')
def predict(body: Body):
    # Tokenize the input and include the attention mask
    tokenized = tokenizer(body.phrase, add_special_tokens=True, return_tensors="pt")
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # Convert inputs to numpy arrays
    inputs = {
        session.get_inputs()[0].name: to_numpy(input_ids),
        session.get_inputs()[1].name: to_numpy(attention_mask),
    }

    #inputs = {
    #   "input_ids": array([[0, 100, 200, ...]]),       # Example NumPy array for input tokens
    #   "attention_mask": array([[1, 1, 1, ...]]),    # Example NumPy array for token mask
    #}


    # Run the model
    out = session.run(None, inputs)

    # Process the output
    result = np.argmax(out)
    return {'positive': bool(result)}

#Run the application using the command: uvicorn main:app --host 0.0.0.0 --port 8000