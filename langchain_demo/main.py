from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

@app.get("/items/x={item_id}&y={item_id2}")
async def read_item(item_id: int, item_id2: int):
    return {"item_id": item_id, "item_id2": item_id2, "result": item_id + item_id2}