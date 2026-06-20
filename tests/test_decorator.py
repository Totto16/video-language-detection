from collections.abc import Callable, Generator
from dataclasses import dataclass
from enum import Enum, IntEnum, StrEnum
from functools import partial, update_wrapper, wraps
from typing import Any, Optional, Self, TypedDict

import pydantic
import pytest
from pytest_subtests import SubTests
from test_helper import re_exact_string

from helper.decorator import decorate_class


def test_decorator_allow_default_works_as_expected() -> None:

    def declare_test() -> None:
        @decorate_class(slots=False, allow_defaults=False)
        class Test1:
            value: str = "default"

            def __init__(self: Self, value: Optional[str] = None) -> None:
                if value is not None:
                    self.value = value

        test1: Any = Test1()

        assert test1.value == "default"

    with pytest.raises(
        TypeError,
        match=re_exact_string("Invalid default value for field 'value': default"),
    ):

        declare_test()


def test_decorator_no_slots_works_as_expected() -> None:

    @decorate_class(slots=False, allow_defaults=True)
    class Test1:
        value: str = "default"

        def __init__(self: Self, value: Optional[str] = None) -> None:
            if value is not None:
                self.value = value

    test1: Any = Test1()

    assert test1.value == "default"

    assert test1.__dict__ == {}

    test1.new_value = 2

    assert test1.new_value == 2

    assert test1.__dict__ == {"new_value": 2}

    test2: Any = Test1("hello")

    assert test2.value == "hello"

    assert test2.__dict__ == {"value": "hello"}

    test2.new_value = 1

    assert test2.new_value == 1

    assert test2.__dict__ == {"value": "hello", "new_value": 1}


def test_decorator_with_slots_works_as_expected() -> None:

    @decorate_class(slots=True, allow_defaults=False)
    class Test1:
        value: str

        def __init__(self: Self, value: str) -> None:
            self.value = value

    test1 = Test1("test1")

    assert test1.value == "test1"

    assert not hasattr(test1, "__dict__")

    test2: Any = Test1("hello")

    assert test2.value == "hello"

    assert not hasattr(test2, "__dict__")

    def add_new_value() -> None:
        test2.new_value = 1

        assert test2.new_value == 1

    with pytest.raises(
        AttributeError,
        match=re_exact_string(
            "'Test1' object has no attribute 'new_value' and no __dict__ for setting new attributes",
        ),
    ):
        add_new_value()

    assert not hasattr(test2, "__dict__")


def test_decorator_edge_cases(  # noqa: PLR0915
    subtests: SubTests,
) -> None:

    with subtests.test("class already defines __slots__"):

        def declare_test() -> None:
            @decorate_class(slots=True, allow_defaults=False)
            class Test1:
                value: str

                __slots__ = ("value",)

                def __init__(self: Self, value: str) -> None:
                    self.value = value

            test1 = Test1("test1")

            assert test1.value == "test1"

        with pytest.raises(
            TypeError,
            match=re_exact_string("Test1 already specifies __slots__"),
        ):
            declare_test()

    with subtests.test("empty class works"):

        @decorate_class(slots=True, allow_defaults=False, weakref_slot=False)
        class TestEmpty:
            pass

        test_empty: Any = TestEmpty()

        assert not hasattr(test_empty, "__dict__")

        assert test_empty.__slots__ == ()

    with subtests.test("parent class has invalid __slots__: case 1"):

        def declare_test() -> None:
            @decorate_class(slots=False, allow_defaults=False)
            class Test1:
                value: str

                __slots__ = lambda x: x * 1  # type: ignore[assignment] # noqa: E731

                def __init__(self: Self, value: str) -> None:
                    self.value = value

            @decorate_class(slots=True, allow_defaults=False)
            class Test2(Test1):
                value2: str

                def __init__(self: Self, value: str, value2: str) -> None:
                    super().__init__(value)

                    self.value2 = value2

            test2 = Test2("test1", "test2")

            assert test2.value == "test1"
            assert test2.value2 == "test2"

        with pytest.raises(
            TypeError,
            match=re_exact_string("'function' object is not iterable"),
        ):
            declare_test()

    with subtests.test("parent class has invalid __slots__: case 2"):

        def declare_test() -> None:
            @decorate_class(slots=False, allow_defaults=False)
            class Test1:
                value: str

                __slots__ = ("value", 1)  # type: ignore[assignment]

                def __init__(self: Self, value: str) -> None:
                    self.value = value

            @decorate_class(slots=True, allow_defaults=False)
            class Test2(Test1):
                value2: str

                def __init__(self: Self, value: str, value2: str) -> None:
                    super().__init__(value)

                    self.value2 = value2

            test2 = Test2("test1", "test2")

            assert test2.value == "test1"
            assert test2.value2 == "test2"

        with pytest.raises(
            TypeError,
            match=re_exact_string("__slots__ items must be strings, not 'int'"),
        ):
            declare_test()

    with subtests.test("parent class has invalid __slots__: case 3"):

        def slot_generator() -> Generator[str | int]:
            yield "value"

            yield 1

        def declare_test() -> None:
            @decorate_class(slots=False, allow_defaults=False)
            class Test1:
                value: str

                __slots__ = slot_generator()  # type: ignore[assignment]

                def __init__(self: Self, value: str) -> None:
                    self.value = value

            @decorate_class(slots=True, allow_defaults=False)
            class Test2(Test1):
                value2: str

                def __init__(self: Self, value: str, value2: str) -> None:
                    super().__init__(value)

                    self.value2 = value2

            test2 = Test2("test1", "test2")

            assert test2.value == "test1"
            assert test2.value2 == "test2"

        with pytest.raises(
            TypeError,
            match=re_exact_string("__slots__ items must be strings, not 'int'"),
        ):
            declare_test()

    with subtests.test("parent class has invalid __slots__: case 4"):

        def declare_test() -> None:
            @decorate_class(slots=False, allow_defaults=False)
            class Test1:
                value: str

                __slots__ = {"value": 1, 2: "int"}  # type: ignore[assignment, dict-item]

                def __init__(self: Self, value: str) -> None:
                    self.value = value

            @decorate_class(slots=True, allow_defaults=False)
            class Test2(Test1):
                value2: str

                def __init__(self: Self, value: str, value2: str) -> None:
                    super().__init__(value)

                    self.value2 = value2

            test2 = Test2("test1", "test2")

            assert test2.value == "test1"
            assert test2.value2 == "test2"

        with pytest.raises(
            TypeError,
            match=re_exact_string("__slots__ items must be strings, not 'int'"),
        ):
            declare_test()


def test_decorator_inheritance(
    subtests: SubTests,
) -> None:
    with subtests.test("parent class test: case 1"):

        @decorate_class(slots=False, allow_defaults=False)
        class Test1:
            value: str

            __slots__ = "value"  # noqa: PLC0205

            def __init__(self: Self, value: str) -> None:
                self.value = value

            def some_call(self: Self) -> int:
                return 1

        @decorate_class(slots=True, allow_defaults=False)
        class Test2(Test1):
            value2: str

            def __init__(self: Self, value: str, value2: str) -> None:
                super().__init__(value)

                self.value2 = value2

            def some_func_using_super(self: Self) -> int:
                return super().some_call() + 1

        test2 = Test2("test1", "test2")

        assert test2.value == "test1"
        assert test2.value2 == "test2"

        assert not hasattr(test2, "__dict__")

        assert test2.some_func_using_super() == 2

    with subtests.test("parent class test: case 2"):

        @decorate_class(slots=False, allow_defaults=False)
        class Test3:
            value: str

            __slots__ = "value"  # noqa: PLC0205

            def __init__(self: Self, value: str) -> None:
                self.value = value

        @decorate_class(slots=True, allow_defaults=False)
        class Test4(Test3):
            value2: str

            def __init__(self: Self, value: str, value2: str) -> None:
                super(Test4, self).__init__(value)  # noqa: UP008

                self.value2 = value2

        test4 = Test4("test3", "test4")

        assert test4.value == "test3"
        assert test4.value2 == "test4"

        assert not hasattr(test4, "__dict__")

    with subtests.test("parent class test: case 3"):

        @decorate_class(slots=False, allow_defaults=False)
        class Test5A:
            value: str

            __slots__ = ("value",)

            def __init__(self: Self, value: str) -> None:
                self.value = value

            def some_call(self: Self) -> int:
                return 1

        @decorate_class(slots=True, allow_defaults=False)
        class Test5B(Test5A):
            value2: str

            def __init__(self: Self, value: str, value2: str) -> None:
                super().__init__(value)

                self.value2 = value2

            def some_func_using_super(self: Self) -> int:
                return super().some_call() + 1

        test5b = Test5B("test5a", "test5b")

        assert test5b.value == "test5a"
        assert test5b.value2 == "test5b"

        assert not hasattr(test2, "__dict__")

        assert test2.some_func_using_super() == 2


# tests from https://github.com/python/cpython/pull/124455/changes#diff-44ce2dc1c4922b2f5cf7631d8f86cc569a4c25eb003aaecdc2bc22eb9163d5f5R1224
def test_decorator_slots_with_super_calls(  # noqa: PLR0915
    subtests: SubTests,
) -> None:
    with subtests.test("test_zero_argument_super"):

        @decorate_class(slots=True)
        class A1:
            def foo(self: Self) -> None:
                super()

        A1().foo()

    with subtests.test("test_dunder_class_with_old_property"):

        @decorate_class(slots=True)
        class A2:
            def _get_foo(self: Self) -> type[A2]:
                assert __class__ is type(self)  # type: ignore[name-defined]
                assert __class__ is self.__class__  # type: ignore[name-defined]
                return __class__  # type: ignore[name-defined]

            def _set_foo(self: Self, value: Any) -> None:  # noqa: ARG002
                assert __class__ is type(self)  # type: ignore[name-defined]
                assert __class__ is self.__class__  # type: ignore[name-defined]

            def _del_foo(self: Self) -> None:
                assert __class__ is type(self)  # type: ignore[name-defined]
                assert __class__ is self.__class__  # type: ignore[name-defined]

            foo = property(_get_foo, _set_foo, _del_foo)

        a = A2()
        assert a.foo is A2
        a.foo = 4
        del a.foo

    with subtests.test("test_dunder_class_with_new_property"):

        @decorate_class(slots=True)
        class A3:
            @property
            def foo(self: Self) -> type[Self]:
                return self.__class__

            @foo.setter
            def foo(self: Self, value: Any) -> None:  # noqa: ARG002
                assert __class__ is type(self)  # type: ignore[name-defined]

            @foo.deleter
            def foo(self: Self) -> None:
                assert __class__ is type(self)  # type: ignore[name-defined]

        a = A3()
        assert a.foo is A3
        a.foo = 4
        del a.foo

    # Test the parts of a property individually.
    with subtests.test("test_slots_dunder_class_property_getter"):

        @decorate_class(slots=True)
        class A4:
            @property
            def foo(self: Self) -> type[A4]:
                return __class__  # type: ignore[name-defined]

        a = A4()
        assert a.foo is A4

    with subtests.test("test_slots_dunder_class_property_setter"):

        @decorate_class(slots=True)
        class A5:
            foo = property()

            @foo.setter  # type: ignore[no-redef]
            def foo(self: Self, val: Any) -> None:  # noqa: ARG002
                assert __class__ is type(self)  # type: ignore[name-defined]

        a = A5()
        a.foo = 4

    with subtests.test("test_slots_dunder_class_property_deleter"):

        @decorate_class(slots=True)
        class A6:
            foo = property()

            @foo.deleter  # type: ignore[no-redef]
            def foo(self: Self) -> None:
                assert __class__ is type(self)  # type: ignore[name-defined]

        a = A6()
        del a.foo

    with subtests.test("test_wrapped"):

        def mydecorator1[T](
            f: Callable[..., T],
        ) -> Callable[..., T]:
            @wraps(f)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                return f(*args, **kwargs)

            return wrapper

        @decorate_class(slots=True)
        class A7:
            @mydecorator1
            def foo(self: Self) -> None:
                super()

        A7().foo()

    with subtests.test("test_remembered_class"):
        # Apply the decorate_class decorator manually (not when the class
        # is created), so that we can keep a reference to the
        # undecorated class.
        class A8:
            def cls(self: Self) -> type[A8]:
                return __class__  # type: ignore[name-defined]

        assert A8().cls() is A8

        B1 = decorate_class(slots=True)(A8)  # noqa: N806
        assert B1().cls() is B1

        # This is undesirable behavior, but is a function of how
        # modifying __class__ in the closure works.  I'm not sure this
        # should be tested or not: I don't really want to guarantee
        # this behavior, but I don't want to lose the point that this
        # is how it works.

        # The underlying class is "broken" by changing its __class__
        # in A.foo() to B.  This normally isn't a problem, because no
        # one will be keeping a reference to the underlying class A.
        assert A8().cls() is B1

    with subtests.test("test_wrapped_property"):

        def mydecorator2[T](
            f: Callable[..., T],
        ) -> Callable[..., T]:
            @wraps(f)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                return f(*args, **kwargs)

            return wrapper

        class B9:
            @property
            def foo(self: Self) -> str:
                return "bar"

        @decorate_class(slots=True)
        class A9(B9):
            @property
            @mydecorator2
            def foo(self: Self) -> str:
                return super().foo

        assert A9().foo == "bar"

    with subtests.test("test_custom_descriptor"):

        class CustomDescriptor1:
            def __init__(self: Self, f: Callable[[A10], Any]) -> None:
                self._f = f

            def __get__(self: Self, instance: A10, owner: Any) -> Any:
                return self._f(instance)

        class B10:
            def foo(self: Self) -> str:
                return "bar"

        @decorate_class(slots=True)
        class A10(B10):
            @CustomDescriptor1
            def foo(cls: Self) -> str:  # noqa: N805
                return super().foo()

        assert A10().foo == "bar"

    with subtests.test("test_custom_descriptor_wrapped"):

        class CustomDescriptor2:
            def __init__(self: Self, f: Callable[[A11], Any]) -> None:
                self._f = update_wrapper(
                    lambda *args, **kwargs: f(*args, **kwargs),  # noqa: PLW0108
                    f,
                )

            def __get__(self: Self, instance: A11, owner: Any) -> Any:
                return self._f(instance)

        class B11:
            def foo(self: Self) -> str:
                return "bar"

        @decorate_class(slots=True)
        class A11(B11):
            @CustomDescriptor2
            def foo(cls: Self) -> str:  # noqa: N805
                return super().foo()

        assert A11().foo == "bar"

    with subtests.test("test_custom_nested_descriptor"):

        class CustomFunctionWrapper3:
            def __init__(self: Self, f: Callable[[A12], Any]) -> None:
                self._f = f

            def __call__(self: Self, *args: Any, **kwargs: Any) -> Any:
                return self._f(*args, **kwargs)

        class CustomDescriptor3:
            def __init__(self: Self, f: Callable[[A12], Any]) -> None:
                self._wrapper = CustomFunctionWrapper3(f)

            def __get__(self: Self, instance: A12, owner: Any) -> Any:
                return self._wrapper(instance)

        class B12:
            def foo(self: Self) -> str:
                return "bar"

        @decorate_class(slots=True)
        class A12(B12):
            @CustomDescriptor3
            def foo(cls: Self) -> str:  # noqa: N805
                return super().foo()

        assert A12().foo == "bar"

    with subtests.test("test_custom_nested_descriptor_with_partial"):

        class CustomDescriptor4:
            def __init__(self: Self, f: Callable[[A13, Any], Any]) -> None:
                self._wrapper: Callable[[A13], Any] = partial(f, value="bar")  # type: ignore[call-arg]

            def __get__(self: Self, instance: A13, owner: Any) -> Any:
                return self._wrapper(instance)

        class B13:
            def foo(self: Self, value: Any) -> Any:
                return value

        @decorate_class(slots=True)
        class A13(B13):
            @CustomDescriptor4
            def foo(self: Self, value: Any) -> Any:
                return super().foo(value)

        assert A13().foo == "bar"

    with subtests.test("test_custom_too_nested_descriptor"):

        class UnnecessaryNestedWrapper5:
            def __init__(self: Self, wrapper: Callable[[A14], Any]) -> None:
                self._wrapper = wrapper

            def __call__(self: Self, *args: Any, **kwargs: Any) -> Any:
                return self._wrapper(*args, **kwargs)

        class CustomFunctionWrapper5:
            def __init__(self: Self, f: Callable[[A14], Any]) -> None:
                self._f = f

            def __call__(self: Self, *args: Any, **kwargs: Any) -> Any:
                return self._f(*args, **kwargs)

        class CustomDescriptor5:
            def __init__(self: Self, f: Callable[[A14], Any]) -> None:
                self._wrapper = UnnecessaryNestedWrapper5(CustomFunctionWrapper5(f))

            def __get__(self: Self, instance: Any, owner: Any) -> Any:
                return self._wrapper(instance)

        class B14:
            def foo(self: Self) -> str:
                return "bar"

        @decorate_class(slots=True)
        class A14(B14):
            @CustomDescriptor5
            def foo(cls: Self) -> str:  # noqa: N805
                return super().foo()

        with pytest.raises(
            TypeError,
            match=re_exact_string(
                "super(type, obj): obj (instance of A14) is not an instance or subtype of type (A14).",
            ),
        ):
            assert A14().foo == "bar"

    with subtests.test("test_user_defined_code_execution"):

        class CustomDescriptor6:
            def __init__(self: Self, f: Callable[[A15, Any], Any]) -> None:
                self._wrapper: Callable[[A15], Any] = partial(f, value="bar")  # type: ignore[call-arg]

            def __get__(self: Self, instance: Any, owner: Any) -> Any:
                return object.__getattribute__(self, "_wrapper")(instance)

            def __getattribute__(self: Self, name: str) -> Any:
                if name in {
                    # these are the bare minimum for the feature to work
                    "__class__",  # accessed on `isinstance(value, Field)`
                    "__wrapped__",  # accessed by unwrap
                    "__get__",  # is required for the descriptor protocol
                    "__dict__",  # is accessed by dir() to work
                }:
                    return object.__getattribute__(self, name)

                msg = f"Never should be accessed: {name}"
                raise RuntimeError(msg)

        class B15:
            def foo(self: Self, value: Any) -> Any:
                return value

        @decorate_class(slots=True)
        class A15(B15):
            @CustomDescriptor6
            def foo(self: Self, value: Any) -> Any:
                return super().foo(value)

        assert A15().foo == "bar"


def test_decorator_invalid_parents(  # noqa: PLR0915
    subtests: SubTests,
) -> None:
    with subtests.test("can't annotate Enum"):

        def declare_test() -> None:
            @decorate_class(slots=True, allow_defaults=False)
            class TestEnum(Enum):
                Value1 = "value1"

        with pytest.raises(
            TypeError,
            match=re_exact_string(
                "Not allowed for class <enum 'TestEnum'> <class 'enum.EnumType'>: An Enum can't be annotated",
            ),
        ):
            declare_test()

    with subtests.test("can't annotate StrEnum"):

        def declare_test() -> None:
            @decorate_class(slots=True, allow_defaults=False)
            class TestEnum(StrEnum):
                Value1 = "value1"

        with pytest.raises(
            TypeError,
            match=re_exact_string(
                "Not allowed for class <enum 'TestEnum'> <class 'enum.EnumType'>: An Enum can't be annotated",
            ),
        ):
            declare_test()

    with subtests.test("can't annotate IntEnum"):

        def declare_test() -> None:
            @decorate_class(slots=True, allow_defaults=False)
            class TestEnum(IntEnum):
                Value1 = 1

        with pytest.raises(
            TypeError,
            match=re_exact_string(
                "Not allowed for class <enum 'TestEnum'> <class 'enum.EnumType'>: An Enum can't be annotated",
            ),
        ):
            declare_test()

    with subtests.test("can't have Enum as parent"):

        def declare_test() -> None:
            class TestEnum(Enum):
                value2 = "value2"

            @decorate_class(slots=True, allow_defaults=False)
            class Test1(TestEnum):  # type: ignore[misc]
                pass

            test1 = Test1("test1")

            assert test1.value == "test1"

        with pytest.raises(
            TypeError,
            match=re_exact_string("<enum 'Test1'> cannot extend <enum 'TestEnum'>"),
        ):
            declare_test()

    with subtests.test("can't annotate TypedDict"):

        def declare_test() -> None:
            @decorate_class(slots=True, allow_defaults=False)
            class TestTypedDict(TypedDict):
                value: str

        with pytest.raises(
            TypeError,
            match=re_exact_string(
                "Not allowed for class <class 'test_decorator.test_decorator_invalid_parents.<locals>.declare_test.<locals>.TestTypedDict'> <class 'typing._TypedDictMeta'>: A TypedDict can't be annotated",
            ),
        ):
            declare_test()

    with subtests.test("can't have TypedDict as parent"):

        def declare_test() -> None:
            class TestTypedDict(TypedDict):
                value2: str

            @decorate_class(slots=True, allow_defaults=False)
            class Test2(TestTypedDict):
                value: str

        with pytest.raises(
            TypeError,
            match=re_exact_string(
                "Not allowed for class <class 'test_decorator.test_decorator_invalid_parents.<locals>.declare_test.<locals>.Test2'> <class 'typing._TypedDictMeta'>: A TypedDict can't be annotated",
            ),
        ):
            declare_test()

    with subtests.test("can't annotate pydantic.BaseModel"):

        def declare_test() -> None:
            @decorate_class(slots=True, allow_defaults=False)
            class TestPydanticBaseModel(pydantic.BaseModel):
                model_config = pydantic.ConfigDict(
                    extra="forbid",
                    strict=True,
                )
                idx: int

        with pytest.raises(
            TypeError,
            match=re_exact_string(
                "Not allowed for class <class 'test_decorator.test_decorator_invalid_parents.<locals>.declare_test.<locals>.TestPydanticBaseModel'> <class 'pydantic._internal._model_construction.ModelMetaclass'>: A Pydantic BaseModel can't be annotated",
            ),
        ):
            declare_test()

    with subtests.test("can't have pydantic.BaseModel as parent"):

        def declare_test() -> None:
            class TestPydanticBaseModel(pydantic.BaseModel):
                model_config = pydantic.ConfigDict(
                    extra="forbid",
                    strict=True,
                )
                idx: int

            @decorate_class(slots=True, allow_defaults=False)
            class Test2(TestPydanticBaseModel):
                value: str

        with pytest.raises(
            TypeError,
            match=re_exact_string(
                "Not allowed for class <class 'test_decorator.test_decorator_invalid_parents.<locals>.declare_test.<locals>.Test2'> <class 'pydantic._internal._model_construction.ModelMetaclass'>: A Pydantic BaseModel can't be annotated",
            ),
        ):
            declare_test()

    with subtests.test("can't annotate Exception"):

        def declare_test() -> None:
            @decorate_class(slots=True, allow_defaults=False)
            class TestException(Exception):  # noqa: N818
                msg2: str

        with pytest.raises(
            TypeError,
            match=re_exact_string(
                "Not allowed for class <class 'test_decorator.test_decorator_invalid_parents.<locals>.declare_test.<locals>.TestException'> <class 'type'>: An Exception can't be annotated",
            ),
        ):
            declare_test()

    with subtests.test("can't have Exception as parent"):

        def declare_test() -> None:
            class TestException(Exception):  # noqa: N818
                msg2: str

            @decorate_class(slots=True, allow_defaults=False)
            class Test2(TestException):
                value: str

        with pytest.raises(
            TypeError,
            match=re_exact_string(
                "Not allowed for class <class 'test_decorator.test_decorator_invalid_parents.<locals>.declare_test.<locals>.Test2'> <class 'type'>: An Exception can't be annotated",
            ),
        ):
            declare_test()

    with subtests.test("can't annotate dataclass"):

        def declare_test() -> None:
            @decorate_class(slots=True, allow_defaults=False)
            @dataclass()
            class TestDataclass:
                msg2: str

        with pytest.raises(
            TypeError,
            match=re_exact_string(
                "Not allowed for class <class 'test_decorator.test_decorator_invalid_parents.<locals>.declare_test.<locals>.TestDataclass'> <class 'type'>: A dataclass can't be annotated",
            ),
        ):
            declare_test()

    with subtests.test("can't annotate dataclass: reversed annotations"):

        def declare_test() -> None:
            @dataclass(slots=True)
            @decorate_class(slots=True, allow_defaults=False)
            class TestDataclass:
                msg2: str

        with pytest.raises(
            TypeError,
            match=re_exact_string(
                "TestDataclass already specifies __slots__",
            ),
        ):
            declare_test()

    with subtests.test("can't annotate dataclass: reversed annotations, no slots"):

        # NOTE: if dataclass runs after the annotation, without slots, we never error out, it would be rather complicated, to implement that
        @dataclass()
        @decorate_class(slots=True, allow_defaults=False)
        class TestDataclass2:
            msg2: str

        test2: TestDataclass2 = TestDataclass2("test")

        assert test2.msg2 == "test"

        assert not hasattr(test2, "__dict__")

    with subtests.test("can't have dataclass as parent"):

        def declare_test() -> None:
            @dataclass()
            class TestDataclass:
                msg2: str

            @decorate_class(slots=True, allow_defaults=False)
            class Test2(TestDataclass):
                value: str

        with pytest.raises(
            TypeError,
            match=re_exact_string(
                "Not allowed for class <class 'test_decorator.test_decorator_invalid_parents.<locals>.declare_test.<locals>.Test2'> <class 'type'>: A dataclass can't be annotated",
            ),
        ):
            declare_test()
