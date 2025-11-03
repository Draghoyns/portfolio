from enum import Enum
from pydantic import BaseModel, HttpUrl, Field


class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


class Image(BaseModel):
    name: str
    url: HttpUrl


class Item(BaseModel):
    name: str = Field(examples=["Sak"])
    description: str | None = Field(examples=["An ambitious AI student"])
    price: float = Field(examples=[49998.99])
    tax: float | None = Field(default=None, examples=[1.0])
    image: Image | None = None


class User(BaseModel):
    username: str
    full_name: str | None = None


class UserIn(BaseModel):
    username: str
    password: str
    full_name: str | None = None


class UserOut(BaseModel):
    username: str
    full_name: str | None = None


class Tags(Enum):
    items = "items"
    users = "users"
