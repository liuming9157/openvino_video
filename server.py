from fastapi import FastAPI
from pydantic import BaseModel
from model import gentext, genimg, tts, removebg
import uvicorn


app = FastAPI()


class Item(BaseModel):
    prompt: str | None = None
    text: str | None = None
    image: str | None = None


@app.get("/")
async def read_root():
    return {"msg": "OpenVINO_video is working!"}


# 文章生成
@app.post("/v1/chat/completions")
async def article(input: Item):
    output = gentext(input.prompt)
    return {"code": 1, "msg": f"{output}!"}


# AI生图
@app.post("/v1/text2image")
async def image(input: Item):
    output = genimg(input.prompt)
    return {"code": 1, "msg": f"{output}!"}


# 文本转语音
@app.post("/v1/tts")
async def voice(input: Item):
    output = tts(input.text)
    return {"code": 1, "msg": f"{output}!"}


# 智能抠图
@app.post("/v1/removebg")
async def koutu(input: Item):
    output = removebg(input.image)
    return {"code": 1, "msg": f"{output}!"}


if __name__ == "__main__":
    uvicorn.run("server:app", port=8080, reload=True)
