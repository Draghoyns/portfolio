from fastapi import FastAPI
from pydantic import BaseModel


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


app = FastAPI()


@app.post("/items/")
async def create_item(item: Item):
    i_dic = item.model_dump()
    if item.tax:
        i_dic.update({"price_with_tax": item.price + item.tax})
    return i_dic


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    return {"id": item_id, **item.model_dump()}
