"""
EPUBKit: A toolkit for analyzing and processing EPUB files
"""

from .parser import (
    HTMLCategoryExtractor,
    CategoryPattern,
    CategoryExtractionError,
    TagInfo,
    CategoryType,
    CategoryMatch,
)

__version__ = "0.1.0"
__all__ = [
    "HTMLCategoryExtractor",
    "CategoryPattern",
    "CategoryExtractionError",
    "TagInfo",
    "CategoryType",
    "CategoryMatch",
    "get_user_tag_examples",
]
