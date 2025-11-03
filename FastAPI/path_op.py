from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from classes import Tags, Item

app = FastAPI()

items = {
    "FF": {"name": "Foo Fighters", "description": "A water stand", "price": 1000},
    "Jolyne": {"name": "Stone Free", "description": "A water stand", "price": 1000},
    "Giorno": {"name": "Gold Experience"},
    "Dio": {"name": "The World"},
    "Kira": {"name": "Killer Queen"},
    "Jotaro": {"name": "Star Platinum"},
}


@app.get("/items/{item_id}", tags=[Tags.items], response_model=Item)
async def read_items(item_id: str):
    return {items[item_id]}


@app.get("/users/", tags=[Tags.users], response_description="A user object")
async def read_user():
    return {"username": "Foo", "fullname": "Foo Fighters"}


@app.put("/items/{item_id}", tags=[Tags.items], response_model=Item)
async def update_item(item_id: str, item: Item):
    updated = jsonable_encoder(item)
    items[item_id] = updated
    return updated
