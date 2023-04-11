# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

# class Question(BaseModel):
#     text: str

# app = FastAPI()

# config = T5Config.from_pretrained("t5-small")
# tokenizer = T5Tokenizer.from_pretrained("t5-small",model_max_length=512)
# model = T5ForConditionalGeneration.from_pretrained("/home/adesoji/nltk_data/RAndom-Nlp-CV-projects/QuestAnswers/t5-small-pytorch", config=config)


# if torch.cuda.is_available():
#     model = model.cuda()

# @app.post("/predict")
# async def predict(question: Question):
#     with torch.no_grad():
#         inputs = tokenizer(question.text, return_tensors="pt")
#         if torch.cuda.is_available():
#             inputs = inputs.to("cuda")
#         outputs = model.generate(**inputs)
#         answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return {"answer": answer}

# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="127.0.0.1", port=8000)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

class QuestionAndPassage(BaseModel):
    question: str
    passage: str

app = FastAPI()

config = T5Config.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small",model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained("QuestandAnswers/t5-small-pytorch", config=config)



if torch.cuda.is_available():
    model = model.cuda()

@app.post("/predict")
async def predict(data: QuestionAndPassage):
    with torch.no_grad():
        input_text = f"question: {data.question} context: {data.passage}"
        inputs = tokenizer(input_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        outputs = model.generate(**inputs)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"answer": answer}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)

#using curl  curl -X POST -H "Content-Type: application/json" -d '{"question": "What is the capital of France?", "passage": "France is a country in Western Europe. Its capital is Paris, which is known for its museums, architectural landmarks, and cafe culture."}' http://localhost:8000/predict
