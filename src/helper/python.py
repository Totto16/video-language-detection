import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def protected(method: F) -> F:
    """
    Decorator to mark a method as protected.
    Allows access only from within the defining class or its subclasses.
    Raises PermissionError if called from outside.
    """

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        stack = inspect.stack()
        # Caller frame: skip current (wrapper) and the method call itself
        caller_frame = stack[2]
        caller_locals = caller_frame.frame.f_locals
        caller_self = caller_locals.get("self", None)

        # Allow same class or subclass access
        if caller_self is not None and isinstance(caller_self, self.__class__):
            return method(self, *args, **kwargs)

        msg = (
            f"Protected method '{method.__name__}' of class "
            f"'{self.__class__.__name__}' cannot be accessed from outside "
            "the class or its subclasses."
        )
        raise PermissionError(msg)

    return cast(F, wrapper)
