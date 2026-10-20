"""Read full-frame RGB from a directory or tar archive without extraction."""
from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
from pathlib import Path, PurePosixPath
from types import TracebackType
import atexit
import hashlib
import tarfile

from PIL import Image

from .schemas import confined


def normalized_relative(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts or ":" in value:
        raise ValueError(f"Invalid relative RGB path: {value}")
    if any(p.casefold() == "apdepth-g-paper" for p in path.parts):
        raise ValueError("Protected directory: ApDepth-G-paper")
    return path.as_posix()


class RGBSource:
    """Archive handles are owned by the caller; only regular files are read."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        if any(p.casefold() == "apdepth-g-paper" for p in self.root.parts):
            raise ValueError("Protected directory: ApDepth-G-paper")
        self.archive: tarfile.TarFile | None = None
        self.members: dict[str, tarfile.TarInfo] = {}
        if self.root.is_dir():
            return
        if not self.root.is_file():
            raise FileNotFoundError(f"RGB directory/archive not found: {self.root}")
        self.archive = tarfile.open(self.root, "r:*")
        try:
            for member in self.archive:
                if not member.isfile():
                    continue
                key = normalized_relative(member.name)
                if key in self.members:
                    raise ValueError(f"Ambiguous duplicate archive member: {key}")
                self.members[key] = member
        except Exception:
            self.close()
            raise

    def contains(self, relative: str) -> bool:
        key = normalized_relative(relative)
        return key in self.members if self.archive else confined(self.root, key).is_file()

    def read(self, relative: str) -> tuple[Image.Image, str]:
        key = normalized_relative(relative)
        if self.archive:
            if key not in self.members:
                raise FileNotFoundError(f"Missing RGB member {key} in {self.root}")
            stream = self.archive.extractfile(self.members[key])
            if stream is None:
                raise ValueError(f"Unreadable RGB member: {key}")
            with stream:
                data = stream.read()
        else:
            data = confined(self.root, key).read_bytes()
        with Image.open(BytesIO(data)) as image:
            # Match the existing loader's pixel coordinates: no EXIF rotation/crop.
            rgb = image.convert("RGB")
            rgb.load()
        return rgb, hashlib.sha256(data).hexdigest()

    def close(self) -> None:
        if self.archive:
            self.archive.close()
            self.archive = None

    def __enter__(self) -> RGBSource:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self.close()


_SOURCES: OrderedDict[tuple[str, int, int], RGBSource] = OrderedDict()


def close_rgb_sources() -> None:
    while _SOURCES:
        _, source = _SOURCES.popitem(last=False)
        source.close()


def read_rgb(root: Path, relative: str) -> tuple[Image.Image, str]:
    """Bounded archive cache; reopen when the archive's size/mtime changes."""
    resolved = root.resolve()
    if resolved.is_dir():
        with RGBSource(resolved) as source:
            return source.read(relative)
    stat = resolved.stat()
    key = (str(resolved), stat.st_mtime_ns, stat.st_size)
    if key not in _SOURCES:
        if len(_SOURCES) >= 3:
            _, previous = _SOURCES.popitem(last=False)
            previous.close()
        _SOURCES[key] = RGBSource(resolved)
    _SOURCES.move_to_end(key)
    return _SOURCES[key].read(relative)


atexit.register(close_rgb_sources)
