import inspect
import itertools
from collections.abc import Callable, Generator
from typing import Optional, cast

# some things here wer copied and modified from the @dataclass annotation


def __get_slots_impl[A](cls: type[A]) -> Generator[str]:
    match cls.__dict__.get("__slots__"):
        # `__dictoffset__` and `__weakrefoffset__` can tell us whether
        # the base type has dict/weakref slots, in a way that works correctly
        # for both Python classes and C extension types. Extension types
        # don't use `__slots__` for slot creation
        case None:
            slots = []
            if getattr(cls, "__weakrefoffset__", -1) != 0:
                slots.append("__weakref__")
            if getattr(cls, "__dictoffset__", -1) != 0:
                slots.append("__dict__")
            yield from slots
        case str(slot):
            yield slot
        # Slots may be any iterable, but we cannot handle an iterator
        # because it will already be (partially) consumed.
        case iterable if not hasattr(iterable, "__next__"):
            yield from iterable
        case _:
            msg = f"Slots of '{cls.__name__}' cannot be determined"
            raise TypeError(msg)


def __add_slots_impl[A](
    cls: type[A],
    field_names: list[str],
    *,
    weakref_slot: bool,
) -> type[A]:
    # Need to create a new class, since we can't set __slots__
    #  after a class has been created.

    # Make sure __slots__ isn't already set.
    if "__slots__" in cls.__dict__:
        msg = f"{cls.__name__} already specifies __slots__"
        raise TypeError(msg)

    # Create a new dict for our new class.
    cls_dict = dict(cls.__dict__)
    field_names = tuple(field_names)
    # Make sure slots don't overlap with those in base classes.
    inherited_slots = set(
        itertools.chain.from_iterable(map(__get_slots_impl, cls.__mro__[1:-1])),
    )
    # The slots for our class.  Remove slots from our base classes.  Add
    # '__weakref__' if weakref_slot was given, unless it is already present.
    cls_dict["__slots__"] = tuple(
        itertools.filterfalse(
            inherited_slots.__contains__,
            itertools.chain(
                # gh-93521: '__weakref__' also needs to be filtered out if
                # already present in inherited_slots
                field_names,
                ("__weakref__",) if weakref_slot else (),
            ),
        ),
    )

    for field_name in field_names:
        # Remove our attributes, if present. They'll still be
        #  available in _MARKER.
        cls_dict.pop(field_name, None)

    # Remove __dict__ itself.
    cls_dict.pop("__dict__", None)

    # Clear existing `__weakref__` descriptor, it belongs to a previous type:
    cls_dict.pop("__weakref__", None)  # gh-102069

    # And finally create the class.
    qualname = getattr(cls, "__qualname__", None)
    cls = type(cls)(cls.__name__, cls.__bases__, cls_dict)
    if qualname is not None:
        cls.__qualname__ = qualname

    return cls


def __process_class_impl[A](
    cls: Optional[type[A]],
    *,
    slots: bool,
    weakref_slot: bool,
    allow_defaults: bool,
) -> type[A]:
    if cls is None:
        msg = "class is None, expected value"
        raise TypeError(msg)

    cls_annotations = inspect.get_annotations(cls)

    if not allow_defaults:
        for field_name in cls_annotations:
            has_field_name = hasattr(cls, field_name)

            if has_field_name:
                msg = f"Invalid default value for field '{field_name}': {getattr(cls, field_name)}"
                raise TypeError(msg)

    if not slots:
        return cls

    field_names: list[str] = list(cls_annotations)

    return __add_slots_impl(cls, field_names, weakref_slot=weakref_slot)


def decorate_class[A](
    *,
    slots: bool = True,
    weakref_slot: bool = True,
    allow_defaults: bool = False,
) -> Callable[[type[A]], type[A]]:

    def wrap(cls: Optional[type[A]]) -> type[A]:
        return __process_class_impl(
            cls,
            slots=slots,
            weakref_slot=weakref_slot,
            allow_defaults=allow_defaults,
        )

    return wrap
