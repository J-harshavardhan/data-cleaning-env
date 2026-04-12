import pandas as pd
from typing import Optional
from .models import DataCleaningObservation, DataCleaningAction

class DataCleaningEnvironment:
    def __init__(self, task_name: str = "remove_duplicates"):
        self.task_name = task_name
        self.df = None
        self.done = False
        self.expected_outliers = 1

    def reset(self, task_name: str = None) -> DataCleaningObservation:
        if task_name:
            self.task_name = task_name
        self.df = pd.DataFrame({
            "name": ["Alice","Bob","Alice","Charlie","Bob"],
            "age": [25, None, 25, 30, None],
            "date": ["2024-01-01","01/02/2024","2024-01-01",
                     "2024-03-01","01/02/2024"],
            "score": [100, 200, 100, 999, 200]
        })
        self.done = False
        return self._get_observation()

    async def reset_async(self, task_name: str = None, **kwargs):
        if task_name:
            self.task_name = task_name
        return self.reset()

    def step(self, action: DataCleaningAction) -> DataCleaningObservation:
        if self.df is None:
            self.reset()

        reward = 0.01
        message = ""

        if action.action_type == "remove_duplicates":
            before = len(self.df)
            self.df = self.df.drop_duplicates()
            removed = before - len(self.df)
            if removed > 0:
                reward = round(min(removed / 3.0, 0.95), 4)
                reward = max(reward, 0.01)
            else:
                reward = 0.01
            message = f"Removed {removed} duplicates"

        elif action.action_type == "fill_missing":
            before = self.df.isnull().sum().sum()
            if before == 0:
                reward = 0.01
                message = "No missing values to fill"
            else:
                col_strategies = {}
                for col in self.df.select_dtypes(include="number").columns:
                    missing = self.df[col].isnull().sum()
                    if missing > 0:
                        skewness = abs(self.df[col].skew())
                        if skewness > 1:
                            self.df[col] = self.df[col].fillna(
                                self.df[col].median()
                            )
                            col_strategies[col] = "median"
                        else:
                            self.df[col] = self.df[col].fillna(
                                self.df[col].mean()
                            )
                            col_strategies[col] = "mean"
                after = self.df.isnull().sum().sum()
                filled = before - after
                fill_ratio = filled / before
                strategy_bonus = 0.10
                reward = round(min(fill_ratio * 0.85 + strategy_bonus, 0.95), 4)
                reward = max(reward, 0.01)
                message = f"Filled {filled} missing values using {col_strategies}"

        elif action.action_type == "fix_outliers":
            q1 = self.df["score"].quantile(0.25)
            q3 = self.df["score"].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (
                (self.df["score"] < lower) |
                (self.df["score"] > upper)
            )
            outliers_found = int(outlier_mask.sum())
            if outliers_found == 0:
                reward = 0.01
                message = "No outliers found"
            else:
                self.df["score"] = self.df["score"].clip(
                    lower=lower, upper=upper
                )
                reward = round(min(outliers_found / 3.0, 0.95), 4)
                reward = max(reward, 0.01)
                message = f"Fixed {outliers_found} outliers (clipped to [{lower:.1f}, {upper:.1f}])"

        self.done = self._check_done()

        # ✅ Return Pydantic model not dict
        obs = self._get_observation()
        obs.reward = reward
        obs.done = self.done
        obs.info = {"message": message}
        return obs

    async def step_async(self, action: DataCleaningAction):
        if self.df is None:
            self.reset()
        return self.step(action)

    def grade(self, task_name: str = None) -> float:
        task = task_name or self.task_name
        if self.df is None:
            self.reset()

        if task == "remove_duplicates":
            dups = self.df.duplicated().sum()
            total = len(self.df)
            score = 1.0 - (dups / total)
            return round(max(0.01, min(score, 0.99)), 4)

        elif task == "fill_missing":
            missing = self.df.isnull().sum().sum()
            total_cells = self.df.size
            score = 1.0 - (missing / total_cells)
            return round(max(0.01, min(score, 0.99)), 4)

        elif task == "fix_outliers":
            q1 = self.df["score"].quantile(0.25)
            q3 = self.df["score"].quantile(0.75)
            iqr = q3 - q1
            outliers = (
                (self.df["score"] < q1 - 1.5 * iqr) |
                (self.df["score"] > q3 + 1.5 * iqr)
            ).sum()
            total = len(self.df)
            score = 1.0 - (outliers / total)
            return round(max(0.01, min(score, 0.99)), 4)

        return 0.5

    def state(self):
        if self.df is None:
            self.reset()
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
            "remove_duplicates": "Remove all duplicate rows from the dataset",
            "fill_missing": "Fill all missing/null values using appropriate strategies (mean/median per column)",
            "fix_outliers": "Detect and fix outlier values in numeric columns using IQR method"
        }
        return tasks.get(self.task_name, "Clean the dataset")

    def _check_done(self) -> bool:
        if self.task_name == "remove_duplicates":
            return bool(self.df.duplicated().sum() == 0)
        elif self.task_name == "fill_missing":
            return bool(self.df.isnull().sum().sum() == 0)
        elif self.task_name == "fix_outliers":
            q1 = self.df["score"].quantile(0.25)
            q3 = self.df["score"].quantile(0.75)
            iqr = q3 - q1
            outliers = (
                (self.df["score"] < q1 - 1.5 * iqr) |
                (self.df["score"] > q3 + 1.5 * iqr)
            ).sum()
            return bool(outliers == 0)
        return False