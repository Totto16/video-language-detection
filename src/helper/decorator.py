import inspect
import itertools
import types
from collections.abc import Callable, Generator
from dataclasses import is_dataclass
from enum import EnumType
from typing import Any, Optional, cast, is_typeddict

from pydantic._internal._model_construction import ModelMetaclass

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
    f: types.FunctionType,
    oldcls: type[A],
    newcls: type[A],
) -> bool:
    try:
        idx = f.__code__.co_freevars.index("__class__")
    except ValueError:
        # This function doesn't reference __class__, so nothing to do.
        return False
    # Fix the cell to point to the new class, if it's already pointing
    # at the old class.  I'm not convinced that the "is oldcls" test
    # is needed, but other than performance can't hurt.

    if f.__closure__ is None:
        msg = "closure should not be none, if co_freevars is set"
        raise TypeError(msg)

    cell = f.__closure__[idx]
    if cell.cell_contents is oldcls:
        cell.cell_contents = newcls
        return True
    return False


_object_members_values = {
    value
    for name, value in (
        *inspect.getmembers_static(object),
        *inspect.getmembers_static(object()),
    )
}


def _is_not_object_member(v: Any) -> bool:
    try:
        return v not in _object_members_values
    except TypeError:
        return True


def _find_inner_functions(
    obj: property,
    seen: Optional[set[int]] = None,
    depth: int = 0,
) -> Generator[types.FunctionType]:
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return None
    seen.add(id(obj))

    depth += 1
    # Normally just an inspection of a descriptor object itself should be enough,
    # and we should encounter the function as its attribute,
    # but in case function was wrapped (e.g. functools.partial was used),
    # we want to dive at least one level deeper.
    if depth > 2:
        return None

    obj_is_type_instance = type in cast(Any, inspect)._static_getmro(  # noqa: SLF001
        type(obj),
    )
    for _, value_iter in inspect.getmembers_static(obj, _is_not_object_member):

        value = value_iter

        value_type = type(value)
        if value_type is types.MemberDescriptorType and not obj_is_type_instance:
            value = value.__get__(obj)
            value_type = type(value)

        if value_type is types.FunctionType:
            yield inspect.unwrap(value)
        else:
            yield from _find_inner_functions(value, seen, depth)


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

    # and also: https://github.com/python/cpython/pull/124692

    newcls: type[A] = cast(type, type(cls))(cls.__name__, cls.__bases__, cls_dict)

    if qualname is not None:
        newcls.__qualname__ = qualname

    # Fix up any closures which reference __class__.  This is used to
    # fix zero argument super so that it points to the correct class
    # (the newly created one, which we're returning) and not the
    # original class.  We can break out of this loop as soon as we
    # make an update, since all closures for a class will share a
    # given cell.  First we try to find a pure function or a property,
    # and then fallback to inspecting custom descriptors
    # if no pure function or property is found.

    custom_descriptors_to_check: list[property] = []
    for member_val in newcls.__dict__.values():
        # If this is a wrapped function, unwrap it.
        member = inspect.unwrap(member_val)

        if isinstance(member, types.FunctionType):
            if __update_func_cell_for__class__impl_(member, cls, newcls):
                break
        elif isinstance(member, property) and (
            any(
                # Unwrap once more in case function
                # was wrapped before it became property.
                __update_func_cell_for__class__impl_(inspect.unwrap(f), cls, newcls)
                for f in (member.fget, member.fset, member.fdel)
                if f is not None
            )
        ):
            break
        elif hasattr(member, "__get__") and not inspect.ismemberdescriptor(member):
            # We don't want to inspect custom descriptors just yet
            # there's still a chance we'll encounter a pure function
            # or a property and won't have to use slower recursive search.
            custom_descriptors_to_check.append(member)
    else:
        # Now let's ensure custom descriptors won't be left out.
        for descriptor in custom_descriptors_to_check:
            for f in _find_inner_functions(descriptor):
                if __update_func_cell_for__class__impl_(f, cls, newcls):
                    break

    return newcls


def __not_allowed_checks_impl[A](cls: type[A]) -> Optional[str]:

    # NOT: don't allow Enums, TypeDicts, pydantic BaseModels, exceptions, dataclasses

    if isinstance(cls, EnumType):
        return "An Enum can't be annotated"  # type: ignore[unreachable]

    if is_typeddict(cls):
        return "A TypedDict can't be annotated"

    if isinstance(cls, ModelMetaclass):
        return "A Pydantic BaseModel can't be annotated"  # type: ignore[unreachable]

    if isinstance(cls, type) and issubclass(cls, Exception):
        return "An Exception can't be annotated"

    if is_dataclass(cls):
        return "A dataclass can't be annotated"

    return None


def __process_class_impl[A](
    cls: type[A],
    *,
    slots: bool,
    weakref_slot: bool,
    allow_defaults: bool,
) -> type[A]:

    not_allowed = __not_allowed_checks_impl(cls)

    if not_allowed is not None:
        msg = f"Not allowed for class {cls} {type(cls)}: {not_allowed}"
        raise TypeError(msg)

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
