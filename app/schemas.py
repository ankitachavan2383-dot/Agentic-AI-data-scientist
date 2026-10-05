import datetime as dt

from pydantic import BaseModel, ConfigDict, Field


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    n_rows: int
    n_cols: int
    created_at: dt.datetime


class RunCreate(BaseModel):
    dataset_id: int
    question: str = Field(min_length=3, max_length=2000)


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    dataset_id: int
    question: str
    status: str
    progress: list = []
    error: str | None = None
    created_at: dt.datetime
    finished_at: dt.datetime | None = None


class SqlQuery(BaseModel):
    query: str
