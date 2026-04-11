import pandas as pd
from typing import Optional
from .models import DataCleaningObservation, DataCleaningAction

class DataCleaningEnvironment:
    def __init__(self, task_name: str = "remove_duplicates"):  # ✅ double underscores        self.task_name = task_name
        self.df = None
        self.done = False
        self.expected_outliers = 1  # score=999 is the outlier

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
            # Partial: 0.5 per duplicate removed, max 1.0
            reward = min(removed / 2, 1.0)
            message = f"Removed {removed} duplicates"

        elif action.action_type == "fill_missing":
            before = self.df.isnull().sum().sum()
            if before == 0:
                reward = 0.0
                message = "No missing values to fill"
            else:
                # Partial credit based on strategy quality
                col_strategies = {}
                for col in self.df.select_dtypes(include="number").columns:
                    missing = self.df[col].isnull().sum()
                    if missing > 0:
                        # Use median for skewed, mean for normal
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
                # Partial: proportion filled + bonus for smart strategy
                fill_ratio = filled / before
                strategy_bonus = 0.2 if len(col_strategies) > 0 else 0.0
                reward = min(fill_ratio + strategy_bonus, 1.0)
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
            outliers_found = outlier_mask.sum()
            
            if outliers_found == 0:
                reward = 0.0
                message = "No outliers found"
            else:
                self.df["score"] = self.df["score"].clip(
                    lower=lower, upper=upper
                )
                # Partial: how many outliers fixed vs expected
                reward = min(outliers_found / self.expected_outliers, 1.0)
                # Bonus if all outliers fixed cleanly
                if outliers_found >= self.expected_outliers:
                    reward = 1.0
                message = f"Fixed {outliers_found} outliers (clipped to [{lower:.1f}, {upper:.1f}])"

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
            "remove_duplicates": "Remove all duplicate rows from the dataset",
            "fill_missing": "Fill all missing/null values using appropriate strategies (mean/median per column)",
            "fix_outliers": "Detect and fix outlier values in numeric columns using IQR method"
        }
        return tasks.get(self.task_name, "Clean the dataset")

    def _check_done(self) -> bool:
        if self.task_name == "remove_duplicates":
            return self.df.duplicated().sum() == 0
        elif self.task_name == "fill_missing":
            return self.df.isnull().sum().sum() == 0
        elif self.task_name == "fix_outliers":
            q1 = self.df["score"].quantile(0.25)
            q3 = self.df["score"].quantile(0.75)
            iqr = q3 - q1
            outliers = (
                (self.df["score"] < q1 - 1.5 * iqr) |
                (self.df["score"] > q3 + 1.5 * iqr)
            ).sum()
            return outliers == 0  # ✅ proper done check
        return False