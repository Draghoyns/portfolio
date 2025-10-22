from fastapi import FastAPI, Cookie
from typing import Annotated
from classes import ModelName

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello, World!"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None):
    if q:
        return {"item_id": item_id, "query": q}
    else:
        return {"item_id": item_id}


@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}


@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}


db = [
    {"name": "Foo"},
    {"name": "Bar"},
    {"name": "Baz"},
    {"name": "Qux"},
    {"name": "Quux"},
    {"name": "Corge"},
    {"name": "Grault"},
    {"name": "Garply"},
    {"name": "Waldo"},
    {"name": "Fred"},
    {"name": "Plugh"},
    {"name": "Xyzzy"},
    {"name": "Thud"},
    {"name": "Lorem"},
    {"name": "Ipsum"},
    {"name": "Dolor"},
    {"name": "Sit"},
    {"name": "Amet"},
]


@app.get("/items/")
async def read_items(skip: int = 0, limit: int = len(db)):
    return db[skip : limit + skip]


@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(
    user_id: int, item_id: str, q: str | None = None, short: bool = False
):
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item.update({"query": q})
    if not short:
        item.update(
            {
                "description": "This item has a very long description because it is amazingly good and perfect. Very recommended item!"
            }
        )
    return item


@app.get("/ads/")
async def read_ads(ads_id: Annotated[str | None, Cookie()] = None):
    return {"ads_id": ads_id}


from classes import UserOut, UserIn


@app.post("/user/", response_model=UserOut)
async def create_user(user: UserIn):
    return user


@app.post("/status/", status_code=204)
async def create_item(name: str):
    return {"name": name}
