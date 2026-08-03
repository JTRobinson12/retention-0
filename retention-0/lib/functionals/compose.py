import functools
from collections.abc import Callable


def _compose_funcs(f: Callable, g: Callable) -> Callable:
    def _composed_fn(*args, **kwargs):
        return g(f(*args, **kwargs))

    return _composed_fn


def compose(*funcs: Callable) -> Callable:
    """Concatenate a list of functions into a single function."""
    return functools.reduce(lambda f, g: _compose_funcs(f, g), funcs)
