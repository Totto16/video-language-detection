import inspect
import functools
from typing import Any, Callable, TypeVar, cast, Union
from helper.types import assert_never

F = TypeVar("F", bound=Callable[..., Any])

C = TypeVar("C", bound=classmethod[Any, Any, Any])
S = TypeVar("S", bound=staticmethod[Any, Any])


def protected(
    member: Union[F, property, C, S],
) -> Union[F, property, C, S]:
    """
    A decorator to mark methods or properties as protected.
    Allows access only from within the defining class or its subclasses.
    Works with instance methods, classmethods, staticmethods, and properties.
    """

    def check_access(self_or_cls: Any) -> None:
        stack = inspect.stack()
        caller_frame = stack[2]
        caller_locals = caller_frame.frame.f_locals
        caller_self = caller_locals.get("self") or caller_locals.get("cls")

        if caller_self is None or not isinstance(caller_self, self_or_cls.__class__):
            raise PermissionError(
                f"Protected member '{getattr(member, '__name__', member)}' "
                f"of class '{self_or_cls.__class__.__name__}' cannot be accessed "
                f"from outside the class or its subclasses."
            )

    # --- Handle normal methods ---
    if callable(member):

        @functools.wraps(member)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            check_access(self)
            return member(self, *args, **kwargs)

        return cast(F, wrapper)

    # --- Handle classmethods ---
    elif isinstance(member, classmethod):
        func = member.__func__

        @classmethod  # type: ignore[misc]
        @functools.wraps(func)
        def class_wrapper(cls: Any, *args: Any, **kwargs: Any) -> Any:
            check_access(cls)
            return func(cls, *args, **kwargs)

        return cast(C, class_wrapper)

    # --- Handle staticmethods ---
    elif isinstance(member, staticmethod):
        func = member.__func__

        @staticmethod  # type: ignore[misc]
        @functools.wraps(func)
        def static_wrapper(*args: Any, **kwargs: Any) -> Any:
            # We can’t infer “self” or “cls” for staticmethods,
            # so protected staticmethods are effectively discouraged
            msg = "Protected staticmethods are not supported."
            raise PermissionError(msg)

        return cast(S, static_wrapper)

    # --- Handle properties ---
    elif isinstance(member, property):
        fget = member.fget
        fset = member.fset
        fdel = member.fdel

        def wrap_accessor(func: Callable[..., Any] | None) -> Callable[..., Any] | None:
            if func is None:
                return None

            @functools.wraps(func)
            def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                check_access(self)
                return func(self, *args, **kwargs)

            return wrapped

        return property(
            fget=wrap_accessor(fget),
            fset=wrap_accessor(fset),
            fdel=wrap_accessor(fdel),
            doc=member.__doc__,
        )
    else:
        assert_never(member)
