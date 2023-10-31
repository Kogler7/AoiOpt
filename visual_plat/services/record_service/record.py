from enum import Enum
from dataclasses import dataclass


class RecordType(Enum):
    reload = 0
    adjust = 1


@dataclass
class RecordUnit:
    layer_tag: str
    record_type: RecordType
    record_data: any


@dataclass
class Record:
    initial: list[RecordUnit]
    updates: list[list[RecordUnit]]
    comp_idx: int  # 兼容编号
