"""Parser interfaces and dispatcher utilities for document ingestion."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
import re
from typing import Any, Iterable, Mapping, MutableSequence, Optional, Protocol, Sequence
from typing import Literal, Tuple, runtime_checkable

from documents.contract_utils import normalize_string, truncate_text

_CONTEXT_TRUNCATION_BYTES = 512
_MAX_SECTION_PATH_LENGTH = 10
_MAX_SECTION_SEGMENT_LENGTH = 128

_BCP47_PATTERN = re.compile(
    r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$"
)


_TEXT_BLOCK_KINDS: Tuple[str, ...] = (
    "paragraph",
    "heading",
    "list",
    "table_summary",
    "slide",
    "note",
    "code",
    "other",
)

TextBlockKind = Literal[_TEXT_BLOCK_KINDS]


def _ensure_non_empty_string(value: str, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(code)
    return value


def _normalise_section_path(section_path: Optional[Iterable[str]]) -> Optional[Tuple[str, ...]]:
    if section_path is None:
        return None
    values: list[str] = []
    for entry in section_path:
        if not isinstance(entry, str):
            raise ValueError("parsed_text_block_section_path")
        normalised = normalize_string(entry)
        if not normalised:
            raise ValueError("parsed_text_block_section_path")
        if len(normalised) > _MAX_SECTION_SEGMENT_LENGTH:
            raise ValueError("parsed_text_block_section_path")
        values.append(normalised)
        if len(values) > _MAX_SECTION_PATH_LENGTH:
            raise ValueError("parsed_text_block_section_path")
    return tuple(values)


def _ensure_optional_int(value: Optional[int], code: str) -> Optional[int]:
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise ValueError(code)
    return value


def _ensure_optional_mapping(
    value: Optional[Mapping[str, Any]], code: str
) -> Optional[Mapping[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(code)
    return MappingProxyType(dict(value))


def _ensure_optional_language(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("parsed_text_block_language")
    if not _BCP47_PATTERN.fullmatch(value):
        raise ValueError("parsed_text_block_language")
    return value


def _ensure_optional_bytes(value: Optional[bytes]) -> Optional[bytes]:
    if value is None:
        return None
    if not isinstance(value, (bytes, bytearray)):
        raise ValueError("parsed_asset_content")
    return bytes(value)


def _ensure_optional_uri(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("parsed_asset_uri")
    return value


def _ensure_optional_bbox(value: Optional[Sequence[float]]) -> Optional[Tuple[float, ...]]:
    """Validate an optional bounding box constrained to the unit square."""
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError("parsed_asset_bbox")
    floats: list[float] = []
    for entry in value:
        if not isinstance(entry, (int, float)):
            raise ValueError("parsed_asset_bbox")
        candidate = float(entry)
        if not isfinite(candidate):
            raise ValueError("parsed_asset_bbox_range")
        floats.append(candidate)
    if len(floats) != 4:
        raise ValueError("parsed_asset_bbox_range")
    x0, y0, x1, y1 = floats
    if not all(0.0 <= coord <= 1.0 for coord in floats):
        raise ValueError("parsed_asset_bbox_range")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("parsed_asset_bbox_range")
    return tuple(floats)


def _ensure_optional_context(value: Optional[str], code: str) -> Optional[str]:
    """Validate optional context strings and truncate them to 512 bytes."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(code)
    truncated = truncate_text(value, _CONTEXT_TRUNCATION_BYTES)
    return truncated


def _ensure_statistics_serializable(value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not isfinite(float(value)):
            raise ValueError("parsed_result_statistics")
        return
    if isinstance(value, str):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _ensure_statistics_serializable(item)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("parsed_result_statistics")
            _ensure_statistics_serializable(item)
        return
    raise ValueError("parsed_result_statistics")


@dataclass(frozen=True)
class ParsedTextBlock:
    """Represents a logical text chunk extracted from a document."""

    text: str
    kind: TextBlockKind
    section_path: Optional[Tuple[str, ...]] = None
    page_index: Optional[int] = None
    table_meta: Optional[Mapping[str, Any]] = None
    language: Optional[str] = None

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        object.__setattr__(self, "text", _ensure_non_empty_string(self.text, "parsed_text_block_text"))
        if self.kind not in _TEXT_BLOCK_KINDS:
            raise ValueError("parsed_text_block_kind")
        object.__setattr__(
            self,
            "section_path",
            _normalise_section_path(self.section_path),
        )
        object.__setattr__(
            self,
            "page_index",
            _ensure_optional_int(self.page_index, "parsed_text_block_page_index"),
        )
        object.__setattr__(
            self,
            "table_meta",
            _ensure_optional_mapping(self.table_meta, "parsed_text_block_table_meta"),
        )
        object.__setattr__(
            self,
            "language",
            _ensure_optional_language(self.language),
        )


@dataclass(frozen=True)
class ParsedTextBlockWithMeta(ParsedTextBlock):
    """Parsed text block carrying optional metadata for downstream consumers."""

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        super().__post_init__()
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ParsedAsset:
    """Represents a non-text asset discovered during parsing."""

    media_type: str
    content: Optional[bytes] = None
    file_uri: Optional[str] = None
    page_index: Optional[int] = None
    bbox: Optional[Tuple[float, ...]] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        object.__setattr__(self, "media_type", _ensure_non_empty_string(self.media_type, "parsed_asset_media_type"))
        object.__setattr__(self, "content", _ensure_optional_bytes(self.content))
        object.__setattr__(self, "file_uri", _ensure_optional_uri(self.file_uri))
        if self.content is None and self.file_uri is None:
            raise ValueError("parsed_asset_location")
        object.__setattr__(
            self,
            "page_index",
            _ensure_optional_int(self.page_index, "parsed_asset_page_index"),
        )
        object.__setattr__(self, "bbox", _ensure_optional_bbox(self.bbox))
        object.__setattr__(
            self,
            "context_before",
            _ensure_optional_context(self.context_before, "parsed_asset_context_before"),
        )
        object.__setattr__(
            self,
            "context_after",
            _ensure_optional_context(self.context_after, "parsed_asset_context_after"),
        )


@dataclass(frozen=True)
class ParsedAssetWithMeta(ParsedAsset):
    """Parsed asset enriched with metadata for downstream stages."""

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        super().__post_init__()
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ParsedResult:
    """Container for parser output prior to persistence."""

    text_blocks: Tuple[ParsedTextBlock, ...] = field(default_factory=tuple)
    assets: Tuple[ParsedAsset, ...] = field(default_factory=tuple)
    statistics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        if not isinstance(self.text_blocks, Sequence):
            raise ValueError("parsed_result_text_blocks")
        object.__setattr__(self, "text_blocks", tuple(self.text_blocks))
        if not all(isinstance(block, ParsedTextBlock) for block in self.text_blocks):
            raise ValueError("parsed_result_text_blocks")

        if not isinstance(self.assets, Sequence):
            raise ValueError("parsed_result_assets")
        object.__setattr__(self, "assets", tuple(self.assets))
        if not all(isinstance(asset, ParsedAsset) for asset in self.assets):
            raise ValueError("parsed_result_assets")

        if not isinstance(self.statistics, Mapping):
            raise ValueError("parsed_result_statistics")
        stats = dict(self.statistics)
        for key, value in stats.items():
            if not isinstance(key, str):
                raise ValueError("parsed_result_statistics")
            _ensure_statistics_serializable(value)
        stats["parse.blocks.total"] = len(self.text_blocks)
        stats["parse.assets.total"] = len(self.assets)
        media_counts = Counter(
            asset.media_type for asset in self.assets if asset.media_type
        )
        for media_type, count in media_counts.items():
            safe_media_type = media_type.replace("/", "_")
            stats[f"parse.assets.media_type.{safe_media_type}"] = count
        object.__setattr__(self, "statistics", stats)


@runtime_checkable
class DocumentParser(Protocol):
    """Protocol that all document parsers must implement."""

    def can_handle(self, document: Any) -> bool:
        """Return True if the parser can handle the given document."""

    def parse(self, document: Any, config: Any) -> ParsedResult:
        """Parse the document and return a ParsedResult."""


class ParserRegistry:
    """Registry maintaining the ordered collection of document parsers."""

    def __init__(self, parsers: Optional[Sequence[DocumentParser]] = None) -> None:
        self._parsers: MutableSequence[DocumentParser] = []
        if parsers:
            for parser in parsers:
                self.register(parser)

    def register(self, parser: DocumentParser, *, prepend: bool = False) -> None:
        if parser is None:
            raise TypeError("parser_invalid")
        can_handle = getattr(parser, "can_handle", None)
        if not callable(can_handle):
            raise TypeError("parser_missing_can_handle")
        parse_method = getattr(parser, "parse", None)
        if not callable(parse_method):
            raise TypeError("parser_missing_parse")
        if not isinstance(parser, DocumentParser):  # type: ignore[arg-type]
            raise TypeError("parser_invalid")
        if prepend:
            self._parsers.insert(0, parser)
        else:
            self._parsers.append(parser)

    @property
    def parsers(self) -> Tuple[DocumentParser, ...]:
        return tuple(self._parsers)

    def dispatch(self, document: Any, config: Any) -> ParsedResult:
        for parser in self._parsers:
            if parser.can_handle(document):
                return parser.parse(document, config)
        raise RuntimeError("no_parser_found")


class ParserDispatcher:
    """Adapter that forwards to a parser registry for compatibility."""

    def __init__(self, registry: ParserRegistry) -> None:
        self._registry = registry

    def parse(self, document: Any, config: Any) -> ParsedResult:
        return self._registry.dispatch(document, config)

