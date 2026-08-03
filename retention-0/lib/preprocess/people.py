import pandas as pd

import assets
import keys
import piping

KEEP = [
    "employee_number",
    "relative_month",
    "month",
    "year",
    "terminated",
    "age",
    "tenure_years",
    "male",
    "ethnicity",
    "rehire",
    "super_or_pm",
    "home_location",
]

SUPERVISOR_ROLES = ("Superintendent", "Project Manager")


def process() -> pd.DataFrame:
    """Explode employees into a person-month panel with a terminal-month target.

    The raw table records terminated at the person level and age/tenure as of
    the extract date; here terminated becomes true only in a person's final
    month, and age and tenure are rolled back to their value in each month.
    """
    return (
        piping.raw_data.read(keys.Imported.PEOPLE)
        .fillna({"employment_end": assets.EXTRACT_DATE})
        .pipe(assets.iso_explode_month, "employment_start", "employment_end", keep_date_col=True)
        .query("month.notna()")
        .astype({"month": "int64", "year": "int64"})
        .assign(
            year_month=lambda _df: assets.calendar_pairs_to_timestamps(_df, "year", "month"),
            relative_month=lambda _df: assets.relative_month(_df, "month", "year"),
            relative_month_max=lambda _df: _df.groupby("employee_number").relative_month.transform("max"),
            terminated=lambda _df: (_df.relative_month == _df.relative_month_max) & _df.terminated,
            age=lambda _df: _df.age - (assets.EXTRACT_DATE - _df.year_month).dt.days / 365.25,
            tenure_years=lambda _df: ((_df.year_month - _df.employment_start).dt.days / 365.25).clip(lower=0),
            super_or_pm=lambda _df: _df.role.isin(SUPERVISOR_ROLES),
        )
        .filter(KEEP)
    )
