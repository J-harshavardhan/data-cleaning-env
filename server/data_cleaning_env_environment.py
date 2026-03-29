import pandas as pd
from typing import Optional
from .models import DataCleaningObservation, DataCleaningAction

class DataCleaningEnvironment:
    def __init__(self, task_name: str = "remove_duplicates"):
        self.task_name = task_name
        self.df = None
        self.done = False

    def reset(self) -> DataCleaningObservation:
        self.df = pd.DataFrame({
            "name": ["Alice","Bob","Alice","Charlie","Bob"],
            "age": [25, None, 25, 30, None],
            "date": ["2024-01-01","01/02/2024","2024-01-01",
                     "2024-03-01","01/02/2024"],
            "score": [100, 200, 100, 999, 200]
        })
        self.done = False
        return self._get_observation()

    async def reset_async(self):
        return self.reset()

    def step(self, action: DataCleaningAction):
        reward = 0.0
        message = ""

        if action.action_type == "remove_duplicates":
            before = len(self.df)
            self.df = self.df.drop_duplicates()
            removed = before - len(self.df)
            reward = min(removed / 2, 1.0)
            message = f"Removed {removed} duplicates"

        elif action.action_type == "fill_missing":
            before = self.df.isnull().sum().sum()
            self.df = self.df.fillna(self.df.mean(numeric_only=True))
            after = self.df.isnull().sum().sum()
            filled = before - after
            reward = min(filled / before, 1.0) if before > 0 else 0.0
            message = f"Filled {filled} missing values"

        elif action.action_type == "fix_outliers":
            q1 = self.df["score"].quantile(0.25)
            q3 = self.df["score"].quantile(0.75)
            iqr = q3 - q1
            outliers = ((self.df["score"] < q1 - 1.5*iqr) |
                       (self.df["score"] > q3 + 1.5*iqr)).sum()
            self.df["score"] = self.df["score"].clip(
                lower=q1 - 1.5*iqr,
                upper=q3 + 1.5*iqr
            )
            reward = 1.0 if outliers > 0 else 0.0
            message = f"Fixed {outliers} outliers"

        self.done = self._check_done()
        return {
            "observation": self._get_observation(),
            "reward": reward,
            "done": self.done,
            "info": {"message": message}
        }

    async def step_async(self, action: DataCleaningAction):
        return self.step(action)

    def state(self):
        return {
            "rows": len(self.df),
            "duplicates": int(self.df.duplicated().sum()),
            "missing": int(self.df.isnull().sum().sum()),
            "task": self.task_name
        }

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def _get_observation(self) -> DataCleaningObservation:
        return DataCleaningObservation(
            dataset_preview=self.df.head().to_string(),
            total_rows=len(self.df),
            duplicate_rows=int(self.df.duplicated().sum()),
            missing_values=int(self.df.isnull().sum().sum()),
            task_description=self._get_task_description()
        )

    def _get_task_description(self) -> str:
        tasks = {
            "remove_duplicates": "Remove all duplicate rows",
            "fill_missing": "Fill all missing/null values",
            "fix_outliers": "Fix outlier values in numeric columns"
        }
        return tasks.get(self.task_name, "Clean the dataset")

    def _check_done(self) -> bool:
        if self.task_name == "remove_duplicates":
            return self.df.duplicated().sum() == 0
        elif self.task_name == "fill_missing":
            return self.df.isnull().sum().sum() == 0
        elif self.task_name == "fix_outliers":
            return True
        return False