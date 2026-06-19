from collections.abc import Generator
import re
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
        with pytest.raises(
            TypeError,
            match=re.escape(
                "super(type, obj): obj (instance of Test2) is not an instance or subtype of type (Test2)."
            ),
        ):

            @decorate_class(slots=False, allow_defaults=False)
            class Test1:
                value: str

                __slots__ = "value"

                def __init__(self: Self, value: str) -> None:
                    self.value = value

            @decorate_class(slots=True, allow_defaults=False)
            class Test2(Test1):
                value2: str

                def __init__(self: Self, value: str, value2: str) -> None:
                    # see: https://github.com/python/cpython/issues/90562
                    # on why this causes issues with slots=True
                    super().__init__(value)

                    self.value2 = value2

            test2 = Test2("test1", "test2")

            assert test2.value == "test1"
            assert test2.value2 == "test2"

            assert not hasattr(test2, "__dict__")

    with subtests.test("parent class test: case 2"):

        @decorate_class(slots=False, allow_defaults=False)
        class Test1:
            value: str

            __slots__ = "value"

            def __init__(self: Self, value: str) -> None:
                self.value = value

        @decorate_class(slots=True, allow_defaults=False)
        class Test2(Test1):
            value2: str

            def __init__(self: Self, value: str, value2: str) -> None:
                super(Test2, self).__init__(value)

                self.value2 = value2

        test2 = Test2("test1", "test2")

        assert test2.value == "test1"
        assert test2.value2 == "test2"

        assert not hasattr(test2, "__dict__")
