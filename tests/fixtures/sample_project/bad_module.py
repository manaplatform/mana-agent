from math import *  # noqa: F403 - intentionally bad fixture import
import os  # noqa: F401 - intentionally unused fixture import
import json  # noqa: F401 - intentionally unused fixture import


def compute(flag: bool) -> int:
    if flag:
        for i in range(2):
            if i > 0:
                while i < 2:
                    if i == 1:
                        return i
    return 0
