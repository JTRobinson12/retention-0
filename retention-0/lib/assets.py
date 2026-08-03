import pandas as pd

NOW = pd.Timestamp.now()
TODAY = pd.Timestamp.today().floor("D")
ONE_WEEK = 7 / 365.25
MINIMUM_YEAR = 2015
EXTRACT_DATE = pd.Timestamp("2026-07-01")
ID_COLUMNS = ["employee_number", "relative_month"]


def iso_range(start_month: int, start_year: int, end_month: int, end_year: int) -> tuple[list, list]:
    """Create a tuple containing a list of months and a list of years to represent every month between two dates.

    Parameters
    ----------
    start_month:
        The calendar month of the first date
    start_year:
        The calendar year of the first date
    end_month:
        The calendar month of the end date
    end_year:
        The calendar year of the end date
    """
    if (end_year < start_year) | ((start_year == end_year) & (end_month < start_month)):
        raise ValueError("Start date must be before end date.")

    current_month, current_year = start_month, start_year
    out_months, out_years = [], []
    stop_month = 12
    while current_year <= end_year:
        if current_year == end_year:
            stop_month = end_month
        remaining_months = range(current_month, stop_month + 1)
        out_months.extend(remaining_months)
        out_years.extend([current_year] * len(remaining_months))
        current_month = 1
        current_year += 1
    return out_months, out_years


def iso_month_year(df: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    """Get month and year from specific columns of type pd.Timestamp from a pd.DataFrame."""
    out = {}
    for date_col in date_columns:
        out[date_col + "_month"], out[date_col + "_year"] = df[date_col].dt.month, df[date_col].dt.year
    return pd.DataFrame(out)


def iso_explode_month(
    df: pd.DataFrame, start_date_col: str, end_date_col: str, keep_date_col: bool = False
) -> pd.DataFrame:
    """Format pd.DataFrame to have one row per calendar month between two dates in each record.

    For samples with invalid date pairs (i.e. one or more missing dates or finish prior to start date),
    we impute a single row where the elements of both month and year are None.

    Parameters
    ----------
    df:
        The pd.DataFrame that contains start and end date columns
    start_date_col:
        A string of the name of the column of type pd.Timestamp which contains start dates
    end_date_col:
        A string of the name of the column of type pd.Timestamp which contains end dates
    keep_col_date:
        A bool that determines if start_date_col and end_date_col should be dropped
    """
    for col in [start_date_col, end_date_col]:
        if not isinstance(col, str):
            raise TypeError(f"{col} must be of type str")
        if col not in df.columns:
            raise ValueError(f"{col} is not a column in df")
    valid_dates = df.query(f"{start_date_col} < {end_date_col}")
    iso_ranges = pd.DataFrame(
        [iso_range(*row[1:]) for row in iso_month_year(valid_dates, [start_date_col, end_date_col]).itertuples()],
        columns=["month", "year"],
        index=valid_dates.index,
    )
    df = (
        df
        .join(iso_ranges)
        .explode(["month", "year"])
        .astype({"month": "Int64", "year": "Int64"})
    )  # fmt: skip
    if not keep_date_col:
        df = df.drop([start_date_col, end_date_col], axis=1)
    return df


def calendar_pairs_to_timestamps(df: pd.DataFrame, year_col: str, month_col: str) -> pd.Series:
    """Convert series of years and months into a series of pandas timestamps."""
    df = df.assign(month=lambda _df: _df[month_col].astype(str).str.rjust(2, "0"))
    return pd.to_datetime(
        pd.Series([f"{y}{m}01" for y, m in zip(df[year_col], df.month)], index=df[year_col].index),
        format="%Y%m%d",
    )


def relative_month(
    df: pd.DataFrame, month_col: str, year_col: str, base_month: int = 1, base_year: int = MINIMUM_YEAR
) -> pd.Series:
    """Get the relative months from a base month."""
    s_months = df[month_col] - base_month
    s_years = df[year_col] - base_year
    return s_months + (12 * s_years)
