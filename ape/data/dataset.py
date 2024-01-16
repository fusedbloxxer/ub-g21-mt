import re
import torch
import torch.utils.data as data
from pathlib import Path
from typing import TypeAlias, Literal, List, Any, Dict, Optional, cast
from collections import defaultdict, namedtuple
from dataclasses import astuple, asdict

from .. import DATA_DIR
from .types import DataSplit
from .types import APETripletPath
from .types import APETripletDict


class APEDataset(data.Dataset):
    def __init__(self, path: Path, split: DataSplit) -> None:
        super(APEDataset, self).__init__()

        # Read dataset
        self.split: DataSplit = split
        self.path: Path = path / split
        self.data: APETripletDict = self.__read_ape_data()

    def __getitem__(self, index) -> Any:
        # Get input data
        src = self.data['src'][index]
        mt = self.data['mt'][index]

        # Get target data
        if 'pe' in self.data and self.data['pe']:
            pe = self.data['pe'][index]
            return dict(src=src, mt=mt, pe=pe)
        return dict(src=src, mt=mt)

    def __len__(self) -> int:
        return len(self.data['src'])

    def __read_ape_data(self) -> APETripletDict:
        # Filter each entry based on string content
        samples: APETripletDict = self.__read_ape_triplets()
        triplets: Dict[str, List[str]] = defaultdict(list)

        # Skip empty strings as they bring no value
        def has_blank(src: str, mt: str, pe: Optional[str]) -> bool:
            return len(src) == 0 or len(mt) == 0 or (pe is not None and len(pe) == 0)

        # Filter data
        for i in range(len(samples['src'])):
            # Skip triplets containing empty strings
            if has_blank(samples['src'][i], samples['mt'][i], samples['pe'][i] if 'pe' in samples and samples['pe'] else None):
                continue

            # Keep Triplet
            triplets['src'].append(samples['src'][i])
            triplets['mt'].append(samples['mt'][i])
            if 'pe' in samples and samples['pe']:
                triplets['pe'].append(samples['pe'][i])
        return cast(APETripletDict, triplets)

    def __read_ape_triplets(self) -> APETripletDict:
        # Store all strings in-memory
        samples: APETripletDict = cast(Any, defaultdict(list))

        # Read files for each triplet at a time
        for triplet in self.__find_ape_files():
            for kind, path in filter(lambda x: x[1], asdict(triplet).items()):
                with open(path, 'r') as file:
                    samples[kind].extend(file.read().splitlines())

        # Perform validations
        if len(samples['src']) != len(samples['mt']):
            raise Exception('src and mt entries must be of the same length')
        if 'pe' in samples and samples['pe'] and len(samples['mt']) != len(samples['pe']):
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
