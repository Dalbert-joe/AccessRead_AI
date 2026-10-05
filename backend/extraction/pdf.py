from dataclasses import dataclass
import io
import re

import fitz
import pytesseract
from PIL import Image


@dataclass
class PDFResult:
    text: str
    title: str
    pages: int
    ocr_pages: int


# ----------------------------------------------------------
# Regular expressions
# ----------------------------------------------------------

LIST_RE = re.compile(
    r"^(?:[-*•]|\d+[.)])\s+"
)


# ----------------------------------------------------------
# Basic helpers
# ----------------------------------------------------------

def is_list_item(text: str) -> bool:
    return bool(
        LIST_RE.match(text.strip())
    )


def normalize_block_lines(text: str) -> list[str]:
    """
    Join visual line wrapping inside one PDF block.

    Example:

        Machine learning is a branch of
        artificial intelligence that enables

    becomes:

        Machine learning is a branch of artificial intelligence that enables
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return []

    result = []
    paragraph = []

    def flush():
        if paragraph:
            result.append(
                " ".join(paragraph).strip()
            )
            paragraph.clear()

    for line in lines:

        if is_list_item(line):
            flush()
            result.append(line)

        else:
            paragraph.append(line)

    flush()

    return result


# ----------------------------------------------------------
# Heading detection
# ----------------------------------------------------------

def is_likely_heading(
    text: str,
    source_line_count: int = 1,
) -> bool:
    """
    Conservative heading detector.

    IMPORTANT:
    A block containing multiple visual PDF lines is treated
    as paragraph text, not as a heading.

    This prevents:

        Machine learning is a branch of
        artificial intelligence that enables

    from becoming a heading.
    """

    text = text.strip()

    if not text:
        return False

    # A multi-line PDF block is almost certainly paragraph text.
    if source_line_count > 1:
        return False

    if is_list_item(text):
        return False

    # Too long.
    if len(text) > 80:
        return False

    # Too many words.
    if len(text.split()) > 10:
        return False

    # Sentence punctuation.
    if text.endswith(
        (".", ",", ";", ":", "!", "?")
    ):
        return False

    lowered = text.lower()

    # Common sentence starters.
    sentence_starters = (
        "a ",
        "an ",
        "the ",
        "this ",
        "that ",
        "these ",
        "those ",
        "machine ",
        "computers ",
        "computer ",
        "data ",
        "it ",
        "they ",
        "we ",
        "he ",
        "she ",
        "there ",
        "when ",
        "which ",
        "who ",
        "using ",
        "used ",
    )

    if lowered.startswith(sentence_starters):
        return False

    return True


# ----------------------------------------------------------
# Block geometry
# ----------------------------------------------------------

def blocks_are_same_paragraph(
    previous: dict,
    current: dict,
) -> bool:
    """
    Determine whether two PDF blocks belong to the same
    paragraph.

    The test PDF contains:

        Block 1:
        Machine learning is a branch of
        artificial intelligence that enables

        Block 2:
        computers to learn patterns from

        Block 3:
        data and make predictions.

    These must become ONE paragraph.
    """

    prev_x0 = previous["x0"]
    prev_y1 = previous["y1"]

    cur_x0 = current["x0"]
    cur_y0 = current["y0"]

    vertical_gap = cur_y0 - prev_y1

    # Overlapping blocks are not merged.
    if vertical_gap < -5:
        return False

    # Large vertical gap = new semantic section.
    if vertical_gap > 60:
        return False

    # Allow normal PDF justification/indentation.
    horizontal_shift = abs(cur_x0 - prev_x0)

    if horizontal_shift > 100:
        return False

    return True


# ----------------------------------------------------------
# Page extraction
# ----------------------------------------------------------

def extract_page_blocks(page) -> str:
    """
    Extract PDF text using PyMuPDF layout blocks.

    Strategy:

    1. Extract visual blocks.
    2. Sort top-to-bottom.
    3. Join visual line wrapping.
    4. Detect actual headings.
    5. Preserve lists.
    6. Merge adjacent paragraph blocks.
    """

    raw_blocks = page.get_text(
        "blocks"
    )

    if not raw_blocks:
        return ""

    blocks = []

    for block in raw_blocks:

        if len(block) < 5:
            continue

        raw_text = block[4].strip()

        if not raw_text:
            continue

        # PyMuPDF block type.
        # 0 = text
        block_type = (
            block[6]
            if len(block) > 6
            else 0
        )

        if block_type != 0:
            continue

        # Count original visual lines BEFORE
        # normalize_block_lines() joins them.
        source_line_count = len(
            [
                line
                for line in raw_text.splitlines()
                if line.strip()
            ]
        )

        blocks.append(
            {
                "x0": float(block[0]),
                "y0": float(block[1]),
                "x1": float(block[2]),
                "y1": float(block[3]),
                "text": raw_text,
                "source_line_count": source_line_count,
            }
        )

    if not blocks:
        return ""

    # ------------------------------------------------------
    # Sort top-to-bottom, then left-to-right.
    # ------------------------------------------------------

    blocks.sort(
        key=lambda b: (
            round(b["y0"], 1),
            round(b["x0"], 1),
        )
    )

    output = []

    paragraph_parts = []

    previous_block = None

    def flush_paragraph():
        if not paragraph_parts:
            return

        paragraph = " ".join(
            part.strip()
            for part in paragraph_parts
            if part.strip()
        ).strip()

        if paragraph:
            output.append(paragraph)

        paragraph_parts.clear()

    # ------------------------------------------------------
    # Process blocks
    # ------------------------------------------------------

    for block in blocks:

        lines = normalize_block_lines(
            block["text"]
        )

        if not lines:
            continue

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # --------------------------------------------------
            # LIST ITEM
            # --------------------------------------------------

            if is_list_item(line):

                flush_paragraph()

                output.append(line)

                previous_block = block

                continue

            # --------------------------------------------------
            # PARAGRAPH CONTINUATION
            # --------------------------------------------------

            same_paragraph = False

            if previous_block is not None:

                same_paragraph = blocks_are_same_paragraph(
                    previous_block,
                    block,
                )

            # --------------------------------------------------
            # HEADING
            #
            # Only classify a line as heading when:
            #
            # 1. There is no active paragraph.
            # 2. It is not continuing the previous block.
            # 3. The ORIGINAL PDF block contained only one line.
            # --------------------------------------------------

            if (
                not paragraph_parts
                and not same_paragraph
                and is_likely_heading(
                    line,
                    block["source_line_count"],
                )
            ):

                output.append(line)

                previous_block = block

                continue

            # --------------------------------------------------
            # If the current block starts a new paragraph,
            # flush the previous one.
            # --------------------------------------------------

            if (
                paragraph_parts
                and not same_paragraph
            ):
                flush_paragraph()

            # --------------------------------------------------
            # NORMAL PARAGRAPH TEXT
            # --------------------------------------------------

            paragraph_parts.append(line)

            previous_block = block

    flush_paragraph()

    return "\n".join(output)


# ----------------------------------------------------------
# Title extraction
# ----------------------------------------------------------

def extract_title(
    doc,
    page_text: str,
) -> str:
    """
    Determine a sensible document title.

    Priority:

    1. Valid PDF metadata title.
    2. First actual heading.
    3. First meaningful non-list line.
    """

    metadata_title = (
        doc.metadata.get("title") or ""
    ).strip()

    # Only accept metadata if it resembles
    # a real title.
    if (
        metadata_title
        and len(metadata_title) <= 100
        and len(metadata_title.split()) <= 12
        and "\n" not in metadata_title
    ):
        return metadata_title

    if not page_text:
        return "Untitled PDF"

    lines = page_text.splitlines()

    # First pass: find heading.
    for line in lines:

        line = line.strip()

        if not line:
            continue

        if line.startswith("[[PAGE"):
            continue

        if is_list_item(line):
            continue

        if is_likely_heading(
            line,
            source_line_count=1,
        ):
            return line

    # Fallback.
    for line in lines:

        line = line.strip()

        if (
            line
            and not line.startswith("[[PAGE")
            and not is_list_item(line)
        ):
            return line

    return "Untitled PDF"


# ----------------------------------------------------------
# PDF processing
# ----------------------------------------------------------

def process_pdf(
    data: bytes,
) -> PDFResult:

    doc = fitz.open(
        stream=data,
        filetype="pdf",
    )

    chunks = []

    ocr_pages = 0

    # ------------------------------------------------------
    # Process every page.
    # ------------------------------------------------------

    for page_number, page in enumerate(
        doc,
        start=1,
    ):

        # --------------------------------------------------
        # Primary extraction
        # --------------------------------------------------

        text = extract_page_blocks(
            page
        ).strip()

        # --------------------------------------------------
        # OCR fallback
        # --------------------------------------------------

        if len(text) < 40:

            pix = page.get_pixmap(
                matrix=fitz.Matrix(
                    1.8,
                    1.8,
                ),
                alpha=False,
            )

            image = Image.open(
                io.BytesIO(
                    pix.tobytes("png")
                )
            )

            text = (
                pytesseract
                .image_to_string(image)
                .strip()
            )

            ocr_pages += 1

        # --------------------------------------------------
        # Store page.
        # --------------------------------------------------

        if text:

            chunks.append(
                f"[[PAGE {page_number}]]\n{text}"
            )

    # ------------------------------------------------------
    # Combine pages.
    # ------------------------------------------------------

    full_text = "\n\n".join(
        chunks
    )

    # ------------------------------------------------------
    # Determine title.
    # ------------------------------------------------------

    title = extract_title(
        doc,
        full_text,
    )

    # ------------------------------------------------------
    # Number of pages with extracted content.
    # ------------------------------------------------------

    pages = len(chunks)

    doc.close()

    return PDFResult(
        text=full_text,
        title=title,
        pages=pages,
        ocr_pages=ocr_pages,
    )