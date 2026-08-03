import json
from dataclasses import dataclass
from typing import Any


@dataclass
class JSON:
    """Convert between python objects and JSON strings."""

    indent: int = 4

    def read(self, key: str) -> Any:
        """Convert a JSON string to its python representation."""
        return json.loads(key)

    def write(self, data: Any, key: str) -> tuple[str, str]:
        """Convert a python object to a JSON-formatted string."""
        return json.dumps(data, indent=self.indent), key
