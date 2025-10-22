from fastapi import FastAPI, Query, Path
from typing import Annotated, Literal
from pydantic import AfterValidator, BaseModel, Field

app = FastAPI()

data = {
    "isbn-9781529046137": "The Hitchhiker's Guide to the Galaxy",
    "imdb-tt0371724": "The Hitchhiker's Guide to the Galaxy",
    "isbn-9781439512982": "Isaac Asimov: The Complete Stories, Vol. 2",
}


def check_valid(s: str):
    if s.startswith(("isbn-", "imdb-")):
        return s
    raise ValueError("Invalid ID format")


@app.get("/items/{id}")
async def read_items(
    id: Annotated[float, Path(title="Item ID", gt=0, lt=10.2)],
    q: Annotated[str | None, Query(alias="item-query")] = None,
):
    results = {"item_id": id}
    if q:
        results.update({"q": q})  # type: ignore
    return results


class FilterParams(BaseModel):
    model_config = {"extra": "forbid"}
    limit: int = Field(100, gt=0, le=100)
    offset: int = Field(0, ge=0)
    order_by: Literal["created_at", "updated_at"] = "created_at"
    tags: set[str] = set()


@app.get("/i/")
async def read_i(filter_q: Annotated[FilterParams, Query()]):
    return filter_q
