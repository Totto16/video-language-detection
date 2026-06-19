from typing import Any, Optional, Self

import pytest

from helper.slots import decorate_class


def test_slots_no_slots_works_as_expected() -> None:

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

    print(test1.__dict__)
    assert test2.__dict__ == {"value": "hello"}

    test2.new_value = 1

    assert test2.new_value == 1

    assert test2.__dict__ == {"value": "hello", "new_value": 1}


def test_slots_inferred_slots_works_as_expected() -> None:

    with pytest.raises(ValueError, match="TODO"):

        @decorate_class(slots=True, allow_defaults=False)
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

    print(test1.__dict__)
    assert test2.__dict__ == {"value": "hello"}

    test2.new_value = 1

    assert test2.new_value == 1

    assert test2.__dict__ == {"value": "hello", "new_value": 1}
