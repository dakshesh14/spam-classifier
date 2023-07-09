import uvicorn

from fastapi import FastAPI
from model.model import Model


model = Model()
app = FastAPI()


@app.get("/predict/{text}")
async def predict(text: str):
    result = model.predict(text)
    return {
        "text": text,
        "label": "spam" if result == 1 else "ham"
    }


if __name__ == "__main__":
    uvicorn.run(
        app="main:app",
        host="localhost",
        port=8000,
        reload=True,
    )
