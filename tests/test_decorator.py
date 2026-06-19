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
