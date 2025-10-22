from fastapi import FastAPI, Body
from typing import Annotated

from classes import Item, User


app = FastAPI()


@app.post("/items/")
async def create_item(item: Item):
    i_dic = item.model_dump()
    if item.tax:
        i_dic.update({"price_with_tax": item.price + item.tax})
    return i_dic


@app.put("/items/{item_id}")
async def update_item_user(
    item_id: int, item: Item, user: User, importance: Annotated[int, Body()]
):

    return {"id": item_id, "item": item, "user": user, "importance": importance}
