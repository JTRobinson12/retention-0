from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Never

import pandas as pd
from plum import (
    Pipeline,
    ReadFromCache,
    core,
    data_source_types,
    mutate,
    pandas_connectors,
    serialize,
    storage,
)


@dataclass
class Builder:
    """Build ML Data.

    EXPERIMENTAL PIPE!
    """

    cache: data_source_types.CompleteDataSource
    processor: Callable[[], pd.DataFrame]
    splitters: dict[str, Callable[[pd.DataFrame], pd.DataFrame]]

    def read(self, key: str) -> pd.DataFrame:
        """Read the raw data and put it into the cache."""
        unsplit_df = self.processor()
        for write_key, splitter in self.splitters.items():
            if write_key != key:
                self.cache.write(splitter(unsplit_df), write_key)
        return self.splitters[key](unsplit_df)

    def write(self, data: Any, key: Any) -> Never:
        """Do not write with this class."""
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        return key in self.splitters


def schema_and_csv_storage(directory: str) -> Pipeline:
    """Store DataFrames as a CSV file alongside a JSON schema of column dtypes."""
    local_schema = mutate.append_key_suffix(
        ".json",
        Pipeline.from_data_sources(
            serialize.JSON(),
            storage.LocalTextFile(f"{directory}/schemas", create_dir=True),
        ),
    )
    local_csv = mutate.append_key_suffix(".csv", storage.LocalTextFile(directory, encoding="utf-8", create_dir=True))
    return Pipeline.from_data_sources(
        pandas_connectors.DataFrameToSchemaAndCSV(),
        core.SideBySide.from_data_sources(local_schema, local_csv),
    )


raw_data = ReadFromCache(storage.Dict(), schema_and_csv_storage("data"))

local_df_storage = schema_and_csv_storage("artifacts/data")

local_csv_cache = ReadFromCache(storage.Dict(), local_df_storage)
