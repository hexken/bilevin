from enum import Enum, IntEnum
from aenum import Enum as AEnum


class TwoDir(IntEnum):
    FORWARD = 0
    BACKWARD = 1


class FourDir(IntEnum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3


class Color(AEnum):

    _init_ = "value string"

    NEUTRAL = 0, "neutral"
    BLUE = 1, "b"
    RED = 2, "r"
    GREEN = 3, "g"
    CYAN = 4, "c"
    YELLOW = 5, "y"
    MAGENTA = 6, "m"
    BLACK = 7, "k"

    def __str__(self):
        return self.string

    @classmethod
    def _missing_value_(cls, value):
        for member in cls:
            if member.string == value:
                return member.value

    @classmethod
    def int_values(cls):
        return [member.value for member in cls]

    @classmethod
    def str_values(cls):
        return [member.string for member in cls]

    def __eq__(self, other):
        if isinstance(other, Color):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        elif isinstance(other, str):
            return self.string == other
