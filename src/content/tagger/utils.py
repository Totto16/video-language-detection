from typing import Literal, assert_never

from helper.utils import dict_has


def merge_dicts[A](
    dict1: dict[str, A],
    dict2: dict[str, A],
    duplicate_behavior: Literal["overwrite", "error", "ignore"],
) -> dict[str, A]:
    res: dict[str, A] = {}
    for key, value in dict1.items():
        res[key] = value  # noqa: PERF403

    for key, value in dict2.items():
        if dict_has(res, key):
            if duplicate_behavior == "error":
                msg = f"Trying to merge duplicate key: {key}"
                raise RuntimeError(msg)

            if duplicate_behavior == "overwrite":
                res[key] = value
            elif duplicate_behavior == "ignore":
                pass
            else:
                assert_never(duplicate_behavior)
        else:
            res[key] = value

    return res
