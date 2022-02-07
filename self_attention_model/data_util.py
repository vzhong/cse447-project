import json
from pathlib import Path
import string
from typing import Dict, Optional


def convert_jsonlist(src_path: Path, dst_path: Path):
    COMMENTS: str = "comments"
    BODY: str = "body"

    with open(src_path, 'r') as src, open(dst_path, 'w') as dst:
        for line in src:
            post = json.loads(line)
            if COMMENTS in post:
                for comment in post[COMMENTS]:
                    if BODY in comment:
                        dst.writelines([comment[BODY].lower()])


class SymbolIndexer:
    _known_symbol_to_index: Dict[str, int]
    _index_to_known_symbol: Dict[int, str]

    _unknown_idx: int
    _size: int

    def _add_symbol(self, symbol: str):
        self._known_symbol_to_index[symbol] = self._size
        self._index_to_known_symbol[self._size] = symbol
        self._size += 1

    def _add_unknown(self):
        self._unknown_idx = self._size
        self._size += 1

    def __init__(self):
        self._known_symbol_to_index = {}
        self._index_to_known_symbol = {}
        self._size = 0

        for symbol in string.ascii_lowercase:
            self._add_symbol(symbol)

        for symbol in string.digits:
            self._add_symbol(symbol)

        self._add_symbol(' ')
        self._add_symbol('.')

        self._add_unknown()

    def size(self) -> int:
        return self._size

    def to_index(self, symbol: str) -> int:
        return self._known_symbol_to_index[symbol] if symbol in self._known_symbol_to_index else self._unknown_idx

    def to_symbol(self, index: int) -> Optional[str]:
        return self._index_to_known_symbol[index] if index in self._index_to_known_symbol else None
