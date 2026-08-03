from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import pandas as pd

Schema = dict[str, str]


@dataclass
class DataFrameToCSVString:
    """Read and write a DataFrame to CSV strings."""

    read_kwargs: dict = field(default_factory=dict)
    write_kwargs: dict = field(default_factory=dict)

    def read(self, key: str) -> pd.DataFrame:
        """Convert a CSV formatted string to a pandas DataFrame."""
        return pd.read_csv(StringIO(key), **self.read_kwargs)

    def write(self, data: pd.DataFrame, key: str) -> tuple[str, str]:
        """Write the data to a string representation of a CSV."""
        return data.to_csv(path_or_buf=None, index=False, **self.write_kwargs), key


@dataclass
class DataFrameToCSVFile:
    """Read and write a DataFrame to a CSV file."""

    read_kwargs: dict = field(default_factory=dict)
    write_kwargs: dict = field(default_factory=dict)

    def read(self, key: str | Path) -> pd.DataFrame:
        """Read data from a CSV file and produce a pandas DataFrame."""
        return pd.read_csv(key, **self.read_kwargs)

    def write(self, data: pd.DataFrame, key: str | Path) -> tuple[None, str | Path]:
        """Write the data to a file at the provided key."""
        return data.to_csv(path_or_buf=key, index=False, **self.write_kwargs), key


@dataclass
class DataFrameToSchemaAndCSV:
    """Convert between a dataframe and a csv-formatted string and dtype schema."""

    read_kwargs: dict = field(default_factory=dict)
    write_kwargs: dict = field(default_factory=dict)

    def read(self, key: tuple[Schema, str]) -> pd.DataFrame:
        """Create a dataframe with schema-provided dtypes from a csv-formatted string."""
        schema, csv_string = key
        date_types, no_dates_schema = [], {}
        for col, dtype in schema.items():
            if "datetime64[ns" in dtype:
                date_types.append(col)
            else:
                no_dates_schema[col] = dtype
        return pd.read_csv(StringIO(csv_string), dtype=no_dates_schema, parse_dates=date_types, **self.read_kwargs)

    def write(self, data: pd.DataFrame, key: str) -> tuple[tuple[Schema, str], str]:
        """Create a dictionary of dtypes and a csv-formatted string from a dataframe.

        None, pd.NA, np.nan, and all other missing values will be written as
        empty strings unless overwritten in write_kwargs.
        """
        schema = {column: str(dtype) for column, dtype in data.dtypes.items()}
        csv_string = data.to_csv(path_or_buf=None, index=False, **self.write_kwargs)
        return (schema, csv_string), key
