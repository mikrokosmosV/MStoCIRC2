"""Missing value imputation – delegates to R native packages."""
from __future__ import annotations
import logging
import pandas as pd
from ..core import DependencyError

log = logging.getLogger(__name__)


def impute(df: pd.DataFrame, method: str) -> pd.DataFrame:
    from .r_runtime_bridge import r_impute, RSCRIPT
    if RSCRIPT is None:
        raise DependencyError(
            "Rscript not found. R is required for DEA imputation. Configure `Rscript` on PATH "
            "or set `MSTOCIRC2_RSCRIPT`, `RSCRIPT`, or `R_HOME`."
        )
    log.info(f"Imputation: {method} via R")
    result = r_impute(df, method)
    if result is None:
        raise DependencyError(
            f"R imputation '{method}' failed. Check the required R packages and your R runtime."
        )
    return result
