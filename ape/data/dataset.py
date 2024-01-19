import re
import torch
import torch.utils.data as data
from pathlib import Path
from typing import TypeAlias, Literal, List, Any, Dict, Optional, cast
from collections import defaultdict, namedtuple
from dataclasses import astuple, asdict

from .. import DATA_DIR
from .types import APESplit
from .types import APETripletPath
from .types import APETripletDict


class APEDataset(data.Dataset[APETripletDict[str]]):
    def __init__(self,
                 path: Path,
                 split: APESplit) -> None:
        super(APEDataset, self).__init__()
        assert split in ('train', 'test')

        # Read dataset
        self.split: APESplit = split
        self.path: Path = path / split
        self.data_raw: APETripletDict[List[str]] = self.__read_ape_data()
        self.data_sources: List[str] = list(self.data_raw.keys())

    def __getitem__(self, index: int) -> APETripletDict[str]:
        context: APETripletDict[str] = cast(Any, {})

        # Get input data
        context['src'] = self.data_raw['src'][index]
        context['mt'] = self.data_raw['mt'][index]

        # Get target data
        if 'pe' in self.data_raw:
            context['pe'] = self.data_raw['pe'][index]

        return context

    def __len__(self) -> int:
        return len(self.data_raw['src'])

    def __read_ape_data(self) -> APETripletDict[List[str]]:
        # Filter each entry based on string content
        samples: APETripletDict[List[str]] = self.__read_ape_triplets()
        triplets: Dict[str, List[str]] = defaultdict(list)

        # Skip empty strings as they bring no value
        def has_blank(src: str, mt: str, pe: Optional[str]) -> bool:
            return len(src) == 0 or len(mt) == 0 or (pe is not None and len(pe) == 0)

        # Filter data
        for i in range(len(samples['src'])):
            # Skip triplets containing empty strings
            if has_blank(samples['src'][i], samples['mt'][i], samples['pe'][i] if 'pe' in samples else None):
                continue

            # Keep Triplet
            triplets['src'].append(samples['src'][i])
            triplets['mt'].append(samples['mt'][i])
            if 'pe' in samples:
                triplets['pe'].append(samples['pe'][i])
        return cast(APETripletDict[List[str]], triplets)

    def __read_ape_triplets(self) -> APETripletDict[List[str]]:
        # Store all strings in-memory
        samples: APETripletDict[List[str]] = cast(Any, defaultdict(list))

        # Read files for each triplet at a time
        for triplet in self.__find_ape_files():
            for kind, path in filter(lambda x: x[1], asdict(triplet).items()):
                with open(path, 'r') as file:
                    samples[kind].extend(file.read().splitlines())

        # Perform validations
        if len(samples['src']) != len(samples['mt']):
            raise Exception('src and mt entries must be of the same length')
        if 'pe' in samples and len(samples['mt']) != len(samples['pe']):
            raise Exception('pe, mt and src entries must be of the same length')
        return samples

    def __find_ape_files(self) -> List[APETripletPath]:
        if not self.path.is_dir():
            raise ValueError(f'{self.path} must be a dir')

        # Store all paths to the triplets
        data_files = defaultdict(dict)

        # Start search from root dir for data
        stack = [self.path]

        # DFS retrieval
        while len(stack) != 0:
            # Expand top-level root
            dirpath = stack.pop()

            # Retrieve in alphabetical order
            for file in sorted(dirpath.iterdir(), key=str, reverse=True):
                if file.is_dir():
                    stack.append(file)
                    continue
                if not re.fullmatch(r'.*\.(mt|pe|src)', file.name):
                    continue
                data_files[file.stem][file.suffix[1:]] = file

        # Construct triplets
        triplets= [APETripletPath(v['src'], v['mt'], v.get('pe', None))
                                        for _, v in data_files.items()]

        # Perform validations
        for triplet in triplets:
            if triplet.src.stem != triplet.mt.stem:
                raise Exception('an APE triplet must have matching src and mt files: {}'.format(triplet))
            if not triplet.pe:
                continue
            if triplet.mt.stem != triplet.pe.stem:
                raise Exception('an APE triplet must have matching mt and pe files: {}'.format(triplet))
        return triplets

    def to_dict(self) -> Dict[str, List[str]]:
        return cast(dict, self.data_raw)

