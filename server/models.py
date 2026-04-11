from pydantic import BaseModel, Field
from typing import Optional

class DataCleaningObservation(BaseModel):
    dataset_preview: str
    total_rows: int
    duplicate_rows: int
    missing_values: int
    task_description: str
    reward: float = 0.01
    done: bool = False
    info: dict = {}

class DataCleaningAction(BaseModel):
    action_type: str
    column: Optional[str] = None
    fill_value: Optional[str] = None

class DataCleaningReward(BaseModel):
    score: float = Field(default=0.5, gt=0.0, lt=1.0)
    message: str = ""