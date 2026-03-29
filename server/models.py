from pydantic import BaseModel
from typing import Optional

class DataCleaningObservation(BaseModel):
    dataset_preview: str
    total_rows: int
    duplicate_rows: int
    missing_values: int
    task_description: str

class DataCleaningAction(BaseModel):
    action_type: str
    column: Optional[str] = None
    fill_value: Optional[str] = None

class DataCleaningReward(BaseModel):
    score: float
    message: str