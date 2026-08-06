import pandas as pd


class BaseStrategy:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def apply_strategy(self) -> pd.DataFrame:
        raise NotImplementedError("Strategy must implement the apply_strategy method")
