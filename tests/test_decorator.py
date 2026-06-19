from collections.abc import Generator
from functools import partial, update_wrapper, wraps
import re
from types import FunctionType
from typing import Any, Optional, Self

import pytest
from pytest_subtests import SubTests

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
        match="Invalid default value for field 'value': default",
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
        match="'Test1' object has no attribute 'new_value' and no __dict__ for setting new attributes",
    ):
        add_new_value()

    assert not hasattr(test2, "__dict__")


def test_decorator_edge_cases(
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
            match="Test1 already specifies __slots_",
        ):
            declare_test()

    with subtests.test("parent class has invalid __slots__: case 1"):

        def declare_test() -> None:
            @decorate_class(slots=False, allow_defaults=False)
            class Test1:
                value: str

                __slots__ = lambda x: x * 1  # noqa: E731

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
            match="'function' object is not iterable",
        ):
            declare_test()

    with subtests.test("parent class has invalid __slots__: case 2"):

        def declare_test() -> None:
            @decorate_class(slots=False, allow_defaults=False)
            class Test1:
                value: str

                __slots__ = ("value", 1)

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
            match="__slots__ items must be strings, not 'int'",
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

                __slots__ = slot_generator()

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
            match="__slots__ items must be strings, not 'int'",
        ):
            declare_test()

    with subtests.test("parent class has invalid __slots__: case 4"):

        def declare_test() -> None:
            @decorate_class(slots=False, allow_defaults=False)
            class Test1:
                value: str

                __slots__ = {"value": 1, 2: "int"}

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
            match="__slots__ items must be strings, not 'int'",
        ):
            declare_test()


def test_decorator_inheritance(
    subtests: SubTests,
) -> None:
    with subtests.test("parent class test: case 1"):

        @decorate_class(slots=False, allow_defaults=False)
        class Test1:
            value: str

            __slots__ = "value"

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

            __slots__ = "value"

            def __init__(self: Self, value: str) -> None:
                self.value = value

        @decorate_class(slots=True, allow_defaults=False)
        class Test4(Test3):
            value2: str

            def __init__(self: Self, value: str, value2: str) -> None:
                super(Test4, self).__init__(value)

                self.value2 = value2

        test4 = Test4("test3", "test4")

        assert test4.value == "test3"
        assert test4.value2 == "test4"

        assert not hasattr(test4, "__dict__")


# tests from https://github.com/python/cpython/pull/124455/changes#diff-44ce2dc1c4922b2f5cf7631d8f86cc569a4c25eb003aaecdc2bc22eb9163d5f5R1224
def test_decorator_slots_with_super_calls(
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
            def _get_foo(self: Self) -> type["A2"]:
                assert __class__ is type(self)
                assert __class__ is self.__class__
                return __class__

            def _set_foo(self: Self, value) -> None:
                assert __class__ is type(self)
                assert __class__ is self.__class__

            def _del_foo(self: Self) -> None:
                assert __class__ is type(self)
                assert __class__ is self.__class__

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
            def foo(self: Self, value) -> None:
                assert __class__ is type(self)

            @foo.deleter
            def foo(self: Self) -> None:
                assert __class__ is type(self)

        a = A3()
        assert a.foo is A3
        a.foo = 4
        del a.foo

    # Test the parts of a property individually.
    with subtests.test("test_slots_dunder_class_property_getter"):

        @decorate_class(slots=True)
        class A4:
            @property
            def foo(self: Self) -> type["A4"]:
                return __class__

        a = A4()
        assert a.foo is A4

    with subtests.test("test_slots_dunder_class_property_setter"):

        @decorate_class(slots=True)
        class A5:
            foo = property()

            @foo.setter
            def foo(self: Self, val) -> None:
                assert __class__ is type(self)

        a = A5()
        a.foo = 4

    with subtests.test("test_slots_dunder_class_property_deleter"):

        @decorate_class(slots=True)
        class A6:
            foo = property()

            @foo.deleter
            def foo(self: Self) -> None:
                assert __class__ is type(self)

        a = A6()
        del a.foo

    with subtests.test("test_wrapped"):

        def mydecorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

            return wrapper

        @decorate_class(slots=True)
        class A7:
            @mydecorator
            def foo(self):
                super()

        A7().foo()

    with subtests.test("test_remembered_class"):
        # Apply the decorate_class decorator manually (not when the class
        # is created), so that we can keep a reference to the
        # undecorated class.
        class A8:
            def cls(self: Self) -> type["A8"]:
                return __class__

        assert A8().cls() is A8

        B1 = decorate_class(slots=True)(A8)
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

        def mydecorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

            return wrapper

        class B9:
            @property
            def foo(self):
                return "bar"

        @decorate_class(slots=True)
        class A9(B9):
            @property
            @mydecorator
            def foo(self):
                return super().foo

        assert A9().foo == "bar"

    with subtests.test("test_custom_descriptor"):

        class CustomDescriptor:
            def __init__(self, f):
                self._f = f

            def __get__(self, instance, owner):
                return self._f(instance)

        class B10:
            def foo(self):
                return "bar"

        @decorate_class(slots=True)
        class A10(B10):
            @CustomDescriptor
            def foo(cls):
                return super().foo()

        assert A10().foo == "bar"

    with subtests.test("test_custom_descriptor_wrapped"):

        class CustomDescriptor:
            def __init__(self, f):
                self._f = update_wrapper(lambda *args, **kwargs: f(*args, **kwargs), f)

            def __get__(self, instance, owner):
                return self._f(instance)

        class B11:
            def foo(self):
                return "bar"

        @decorate_class(slots=True)
        class A11(B11):
            @CustomDescriptor
            def foo(cls):
                return super().foo()

        assert A11().foo == "bar"

    with subtests.test("test_custom_nested_descriptor"):

        class CustomFunctionWrapper:
            def __init__(self, f):
                self._f = f

            def __call__(self, *args, **kwargs):
                return self._f(*args, **kwargs)

        class CustomDescriptor:
            def __init__(self, f):
                self._wrapper = CustomFunctionWrapper(f)

            def __get__(self, instance, owner):
                return self._wrapper(instance)

        class B12:
            def foo(self):
                return "bar"

        @decorate_class(slots=True)
        class A12(B12):
            @CustomDescriptor
            def foo(cls):
                return super().foo()

        assert A12().foo == "bar"

    with subtests.test("test_custom_nested_descriptor_with_partial"):

        class CustomDescriptor:
            def __init__(self, f):
                self._wrapper = partial(f, value="bar")

            def __get__(self, instance, owner):
                return self._wrapper(instance)

        class B13:
            def foo(self, value):
                return value

        @decorate_class(slots=True)
        class A13(B13):
            @CustomDescriptor
            def foo(self, value):
                return super().foo(value)

        assert A13().foo == "bar"

    with subtests.test("test_custom_too_nested_descriptor"):

        class UnnecessaryNestedWrapper:
            def __init__(self, wrapper):
                self._wrapper = wrapper

            def __call__(self, *args, **kwargs):
                return self._wrapper(*args, **kwargs)

        class CustomFunctionWrapper:
            def __init__(self, f):
                self._f = f

            def __call__(self, *args, **kwargs):
                return self._f(*args, **kwargs)

        class CustomDescriptor:
            def __init__(self, f):
                self._wrapper = UnnecessaryNestedWrapper(CustomFunctionWrapper(f))

            def __get__(self, instance, owner):
                return self._wrapper(instance)

        class B14:
            def foo(self):
                return "bar"

        @decorate_class(slots=True)
        class A14(B14):
            @CustomDescriptor
            def foo(cls):
                return super().foo()

        with pytest.raises(
            TypeError,
            match=re.escape(
                "super(type, obj): obj (instance of A14) is not an instance or subtype of type (A14)."
            ),
        ):
            A14().foo

    with subtests.test("test_user_defined_code_execution"):

        class CustomDescriptor:
            def __init__(self, f):
                self._wrapper = partial(f, value="bar")

            def __get__(self, instance, owner):
                return object.__getattribute__(self, "_wrapper")(instance)

            def __getattribute__(self, name):
                if name in {
                    # these are the bare minimum for the feature to work
                    "__class__",  # accessed on `isinstance(value, Field)`
                    "__wrapped__",  # accessed by unwrap
                    "__get__",  # is required for the descriptor protocol
                    "__dict__",  # is accessed by dir() to work
                }:
                    return object.__getattribute__(self, name)
                raise RuntimeError(f"Never should be accessed: {name}")

        class B15:
            def foo(self, value):
                return value

        @decorate_class(slots=True)
        class A15(B15):
            @CustomDescriptor
            def foo(self, value):
                return super().foo(value)

        assert A15().foo == "bar"

        @decorate_class(slots=True)
        class A15(B15):
            @CustomDescriptor
            def foo(self, value):
                return super().foo(value)

        assert A15().foo == "bar"
