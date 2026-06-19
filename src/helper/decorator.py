import inspect
import itertools
import types
from collections.abc import Callable, Generator
from typing import Optional

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


def __update_func_cell_for__class__impl_[A](
    f: Optional[types.FunctionType],
    oldcls: type[A],
    newcls: type[A],
) -> bool:
    # Returns True if we update a cell, else False.
    if f is None:
        # f will be None in the case of a property where not all of
        # fget, fset, and fdel are used.  Nothing to do in that case.
        return False
    try:
        idx = f.__code__.co_freevars.index("__class__")
    except ValueError:
        # This function doesn't reference __class__, so nothing to do.
        return False
    # Fix the cell to point to the new class, if it's already pointing
    # at the old class.  I'm not convinced that the "is oldcls" test
    # is needed, but other than performance can't hurt.
    closure = f.__closure__[idx]
    if closure.cell_contents is oldcls:
        closure.cell_contents = newcls
        return True
    return False


def __add_slots_impl[A](
    cls: type[A],
    field_names: list[str],
    *,
    weakref_slot: bool,
) -> type[A]:
    # Need to create a new class, since we can't set __slots__ after a
    # class has been created, and the @dataclass decorator is called
    # after the class is created.

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

    # see: https://github.com/python/cpython/issues/90562
    # and: https://bugs.python.org/issue46404
    # on why this is needed, when replacing the old class

    # and: https://github.com/python/cpython/pull/124455
    # for the solution

    newcls: type[A] = type(cls)(cls.__name__, cls.__bases__, cls_dict)

    if qualname is not None:
        newcls.__qualname__ = qualname

    # Fix up any closures which reference __class__.  This is used to
    # fix zero argument super so that it points to the correct class
    # (the newly created one, which we're returning) and not the
    # original class.  We can break out of this loop as soon as we
    # make an update, since all closures for a class will share a
    # given cell.
    for member in newcls.__dict__.values():
        # If this is a wrapped function, unwrap it.
        member = inspect.unwrap(member)

        if isinstance(member, types.FunctionType):
            if __update_func_cell_for__class__impl_(member, cls, newcls):
                break
        elif isinstance(member, property):
            if (
                __update_func_cell_for__class__impl_(member.fget, cls, newcls)
                or __update_func_cell_for__class__impl_(member.fset, cls, newcls)
                or __update_func_cell_for__class__impl_(member.fdel, cls, newcls)
            ):
                break

    return newcls


def __process_class_impl[A](
    cls: type[A],
    *,
    slots: bool,
    weakref_slot: bool,
    allow_defaults: bool,
) -> type[A]:
    cls_annotations = inspect.get_annotations(cls)

    if not allow_defaults:
        for field_name in cls_annotations:
            has_field_name = hasattr(cls, field_name)

            if has_field_name:
                field_val = getattr(cls, field_name)

                if isinstance(field_val, types.MemberDescriptorType):
                    # descriptors are used by slots, which is later checked to not be already defined
                    continue

                msg = f"Invalid default value for field '{field_name}': {field_val}"
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

    def wrap(cls: type[A]) -> type[A]:
        return __process_class_impl(
            cls,
            slots=slots,
            weakref_slot=weakref_slot,
            allow_defaults=allow_defaults,
        )

    return wrap
