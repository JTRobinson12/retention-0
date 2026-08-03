import os
from collections.abc import Hashable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Dict:
    """Dictionary data source."""

    data: dict[Hashable, Any] = field(default_factory=dict)

    def read(self, key: Hashable) -> Any:
        """Read a key from the dictionary, if it exists."""
        return self.data[key]

    def write(self, data: Any, key: Hashable) -> tuple[None, Hashable]:
        """Write data into the dictionary."""
        self.data[key] = data
        return None, key

    def exists(self, key: Hashable) -> bool:
        """Check if a key is in the dict."""
        return key in self.data


@dataclass
class LocalTextFile:
    """Data source for text files stored locally."""

    path: str | os.PathLike = ""
    encoding: str | None = None
    create_dir: bool = False

    def read(self, key: str) -> str:
        """Read text from a local file."""
        with Path.open(Path(self.path, key), "r", encoding=self.encoding) as f:
            return f.read()

    def write(self, s: str, key: str) -> tuple[int, str]:
        """Write text to a local file."""
        if self.create_dir:
            Path(self.path).mkdir(parents=True, exist_ok=True)
        with Path.open(Path(self.path, key), "w", encoding=self.encoding) as f:
            out = f.write(s)
        return out, key

    def exists(self, key: str) -> bool:
        """Check if a file exists at path + key."""
        return Path(self.path, key).is_file()


@dataclass
class LocalBinaryFile:
    """Data source for writing binary files locally."""

    path: str | os.PathLike = ""
    create_dir: bool = False

    def read(self, key: str) -> bytes:
        """Read a bytestring from a local file."""
        with Path.open(Path(self.path, key), "rb") as f:
            return f.read()

    def write(self, data: bytes, key: str) -> tuple[int, str]:
        """Write a bytestring to local storage."""
        if self.create_dir:
            Path(self.path).mkdir(parents=True, exist_ok=True)
        with Path.open(Path(self.path, key), "wb") as f:
            out = f.write(data)
        return out, key

    def exists(self, key: str) -> bool:
        """Check if a file exists at path + key."""
        return Path(self.path, key).is_file()
