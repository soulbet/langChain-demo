from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl

app = FastAPI()

class Image(BaseModel):
    url: HttpUrl    
    name: str

@app.post("/images/multiple/")
async def create_multiple_images(images: list[Image]):
    return images


@app.post("/index-weights/")
async def create_index_weights(weights: dict[int, float]):
    return weights