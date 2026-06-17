from typing import Any, Literal, assert_never


def merge_dicts(
    dict1: dict[str, Any],
    dict2: dict[str, Any],
    duplicate_behavior: Literal["overwrite", "error", "ignore"],
) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for key, value in dict1.items():
        res[key] = value  # noqa: PERF403

    for key, value in dict2.items():
        if res.get(key, None) is not None:  # noqa: SIM910
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
