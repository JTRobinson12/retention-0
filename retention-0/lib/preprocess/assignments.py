import pandas as pd

import assets
import keys
import piping

KEEP = [
    "project_number",
    "employee_number",
    "relative_month",
    "employee_project_month",
]


def process() -> pd.DataFrame:
    """Explode assignments into an employee-project-month panel.

    When stints overlap within a calendar month, keep the project the
    employee joined most recently so the panel stays one row per
    employee-month.
    """
    return (
        piping.raw_data.read(keys.Imported.ASSIGNMENTS)
        .pipe(assets.iso_explode_month, "assignment_start", "assignment_end", keep_date_col=True)
        .query("month.notna()")
        .assign(relative_month=lambda _df: assets.relative_month(_df, "month", "year"))
        .sort_values(["employee_number", "project_number", "relative_month"])
        .drop_duplicates(["employee_number", "project_number", "relative_month"])
        .assign(employee_project_month=lambda _df: _df.groupby(["employee_number", "project_number"]).cumcount() + 1)
        .sort_values(["employee_number", "relative_month", "assignment_start"])
        .drop_duplicates(["employee_number", "relative_month"], keep="last")
        .filter(KEEP)
    )
