from collections.abc import Callable
from types import MethodType
from typing import Any, TypeVar

from . import data_source_types

KeyType = TypeVar("KeyType")
ProvidedDataSource = TypeVar("ProvidedDataSource")


def change_key(
    mutator: Callable[[KeyType], KeyType],
    data_source: ProvidedDataSource,
) -> ProvidedDataSource:
    """Mutate a data source to transform a key before any operation."""
    if isinstance(data_source, data_source_types.Readable):

        def mutated_read(self, key: KeyType) -> Any:
            """Read a MUTATED key from storage."""
            return self.__class__.read(self, mutator(key))

        data_source.read = MethodType(mutated_read, data_source)

    if isinstance(data_source, data_source_types.Writeable):

        def mutated_write(self, data: Any, key: KeyType) -> tuple[Any, KeyType]:
            """Write to storage with a MUTATED key."""
            return self.__class__.write(self, data, mutator(key))

        data_source.write = MethodType(mutated_write, data_source)

    if isinstance(data_source, data_source_types.Stateful):

        def mutated_exists(self, key: KeyType) -> bool:
            """Check if a MUTATED key exists."""
            return self.__class__.exists(self, mutator(key))

        data_source.exists = MethodType(mutated_exists, data_source)

    return data_source


def append_key_suffix(suffix: str, data_source: ProvidedDataSource) -> ProvidedDataSource:
    """Append a suffix to a string key before all operations."""
    return change_key(lambda s: s + suffix, data_source)
