from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Readable(Protocol):
    """Protocol for data sources that support read operations."""

    def read(self, key: Any) -> Any:
        """Perform a read operation."""


@runtime_checkable
class Writeable(Protocol):
    """Protocol for data sources that support write operations."""

    def write(self, data: Any, key: Any) -> tuple[Any, Any]:
        """Perform a write operation."""


@runtime_checkable
class Stateful(Protocol):
    """Protocol for data sources that support exist checks."""

    def exists(self, key: Any) -> bool:
        """Check for the existance of a key."""


@runtime_checkable
class Bidirection(Readable, Writeable, Protocol):
    """Protocol for data sources that support reads and writes."""


@runtime_checkable
class CompleteDataSource(Stateful, Bidirection, Protocol):
    """Protocol for data sources supporting reading, writing, and checking state."""


DataSource = Readable | Writeable | Stateful
