from dataclasses import dataclass
from typing import Any

from . import data_source_types


@dataclass
class MultiDataSource:
    """Base class for data sources that call multiple data sources in each step."""

    steps: tuple[tuple[str | int, data_source_types.DataSource], ...]

    @classmethod
    def from_dict(cls, d: dict[str, data_source_types.DataSource]):
        """Create a new instance from a dictionary."""
        return cls(tuple((k, v) for k, v in d.items()))

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Create a pipeline from keyword arguments."""
        return cls(tuple((k, v) for k, v in kwargs.items()))

    @classmethod
    def from_data_sources(cls, *args):
        """Create a pipeline without names from datasources."""
        return cls(tuple((i, ds) for i, ds in enumerate(args)))


@dataclass
class Pipeline(MultiDataSource):
    """Call multiple data sources sequentially."""

    def read(self, key: Any) -> Any:
        """Read a key from the datasource pipeline."""
        for _, ds in reversed(self.steps):
            if isinstance(ds, data_source_types.Readable):
                key = ds.read(key)
        return key

    def write(self, data: Any, key: Any) -> tuple[Any, Any]:
        """Write a key from the datasource pipeline."""
        for _, ds in self.steps:
            if isinstance(ds, data_source_types.Writeable):
                data, key = ds.write(data, key)
        return data, key

    def exists(self, key: Any) -> bool:
        """Check if a key exists."""
        out, ds_iter = True, (step[1] for step in reversed(self.steps))
        for ds in ds_iter:
            if isinstance(ds, data_source_types.Stateful):
                out = ds.exists(key)
            if not out:
                break
        return out


@dataclass
class SideBySide(MultiDataSource):
    """Call multiple datasources in parallel and combine the results."""

    def read(self, key: Any) -> tuple[Any, ...]:
        """Read a key from each datasource and combine the results in a tuple."""
        return tuple(ds.read(key) for _, ds in self.steps if isinstance(ds, data_source_types.Readable))

    def write(self, data: tuple[Any, ...], key: Any) -> tuple[tuple[Any, ...], Any]:
        """Write data to each datasource."""
        writeable_sources = [ds for _, ds in self.steps if isinstance(ds, data_source_types.Writeable)]
        written_data = tuple(ws.write(d, key)[0] for d, ws in zip(data, writeable_sources))
        return written_data, key

    def exists(self, key: Any) -> bool:
        """Check if a key exists in ALL component stateful datasources."""
        return all(ds.exists(key) for _, ds in self.steps if isinstance(ds, data_source_types.Stateful))


@dataclass
class ReadFromCache:
    """Look for key in cache before checking elsewhere."""

    cache: data_source_types.CompleteDataSource
    base: data_source_types.CompleteDataSource

    def read(self, key: Any) -> Any:
        """Read a key from a cache, if possible. Otherwise, get it from the base."""
        if self.cache.exists(key):
            return self.cache.read(key)
        out = self.base.read(key)
        self.cache.write(out, key)
        return out

    def write(self, data: Any, key: Any) -> tuple[Any, Any]:
        """Write the data to the base."""
        return self.base.write(data, key)

    def exists(self, key: Any) -> bool:
        """Check if a key exists."""
        return self.cache.exists(key) or self.base.exists(key)
