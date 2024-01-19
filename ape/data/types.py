from typing import Literal, TypeAlias, TypedDict, List, Optional, Generic, TypeVar, NotRequired
from dataclasses import dataclass
from pathlib import Path

T_APE = TypeVar('T_APE')


APESplit: TypeAlias = Literal[
    'train',
    'test',
]


DataSplit: TypeAlias = Literal[
    'train',
    'valid',
    'test',
]


@dataclass
class APETripletPath:
    src: Path
    mt: Path
    pe: Optional[Path]


class APETripletDict(Generic[T_APE], TypedDict):
    src: T_APE
    mt: T_APE
    pe: NotRequired[T_APE]
