import pandas as pd

import assets
import keys
import piping

KEEP = [
    "project_number",
    "relative_month",
    "project_month",
    "location",
    "priority_project",
    "budget",
]


def process() -> pd.DataFrame:
    """Explode projects into a project-month panel with an elapsed-month counter."""
    return (
        piping.raw_data.read(keys.Imported.PROJECTS)
        .pipe(assets.iso_explode_month, "project_start", "project_end")
        .query("month.notna()")
        .assign(relative_month=lambda _df: assets.relative_month(_df, "month", "year"))
        .sort_values(["project_number", "relative_month"])
        .assign(project_month=lambda _df: _df.groupby("project_number").cumcount() + 1)
        .filter(KEEP)
    )
