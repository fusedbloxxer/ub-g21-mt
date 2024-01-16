from typing import Literal, TypeAlias, TypedDict, List, Optional
from dataclasses import dataclass
from pathlib import Path


DataSplit: TypeAlias = Literal[
    'train',
    'test',
]


@dataclass
class APETripletPath:
    src: Path
    mt: Path
    pe: Optional[Path]


class APETripletDict(TypedDict):
    src: List[str]
    mt: List[str]
    pe: Optional[List[str]]
