from fastapi import FastAPI, Form, HTTPException
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException

from typing import Annotated
from pydantic import BaseModel

app = FastAPI()


class FormData(BaseModel):
    username: str
    password: str


@app.post("/login/")
async def login(data: Annotated[FormData, Form()]):
    return {"username": data.username}


items = {
    "FF": "Foo Fighters",
    "Jolyne": "Stone Free",
    "Giorno": "Gold Experience",
    "Dio": "The World",
    "Kira": "Killer Queen",
    "Jotaro": "Star Platinum",
}


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc):
    print(f"OMG! An HTTP error!: {repr(exc)}")
    return await http_exception_handler(request, exc)


@app.get("/items/{item_id}")
async def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(
            status_code=418,
            detail="Stand user not found",
        )

    return {"Stand user": item_id, "Stand": items[item_id]}
