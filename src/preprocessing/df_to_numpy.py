from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

def df_to_numpy(
    df: pd.DataFrame,
    tic_list: List[str],
    date_sorted: bool = True,
    tech_indicator_list: Optional[List[str]] = None,
    price_col: str = "close",
    turbulence_col: Optional[str] = "turbulence",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a DataFrame to numpy arrays:
      - price_array (T, N)
      - tech_array (T, N, F)
      - turbulence_array (T,)
      - dates (T,)
    """

    if not date_sorted:
        df = df.sort_values("date")

    price_pivot = df.pivot(index="date", columns="tic", values=price_col)
    price_pivot = price_pivot.reindex(columns=tic_list)
    price_pivot = price_pivot.ffill().bfill().fillna(0.0)

    dates = price_pivot.index.tolist()
    price_array = price_pivot.values.astype(np.float32)

    if tech_indicator_list:
        tech_list = []

        for ind in tech_indicator_list:
            pivot = df.pivot(index="date", columns="tic", values=ind)
            pivot = pivot.reindex(columns=tic_list)
            pivot = pivot.ffill().bfill().fillna(0.0)
            tech_list.append(pivot.values.astype(np.float32))

        tech_array = np.stack(tech_list, axis=2)
    else:
        tech_array = np.zeros((price_array.shape[0], price_array.shape[1], 0), dtype=np.float32)


    if turbulence_col and turbulence_col in df.columns:
        turb_series = (df
                       .drop_duplicates(subset=["date"])
                       .set_index("date")[turbulence_col]
                       )
        turb_series = turb_series.reindex(index=price_pivot.index).fillna(0.0)
        turbulence_array = turb_series.values.astype(np.float32)
    else:
        turbulence_array = np.zeros(price_array.shape[0], dtype=np.float32)

    return price_array, tech_array, turbulence_array, np.array(dates)
