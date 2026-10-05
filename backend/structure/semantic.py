import re
from inference.runtime import get_inference_engine


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

LIST_RE = re.compile(
    r"^(?:[-*•]|\d+[.)])\s+(.+)$"
)


# Words that are commonly used as standalone section headings.
COMMON_HEADINGS = {
    "introduction",
    "abstract",
    "background",
    "overview",
    "applications",
    "features",
    "advantages",
    "disadvantages",
    "methodology",
    "methods",
    "implementation",
    "architecture",
    "results",
    "discussion",
    "analysis",
    "evaluation",
    "conclusion",
    "references",
    "summary",
    "future work",
}


def clean_lines(text):
    text = re.sub(r"\r", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return [
        x.strip()
        for x in text.splitlines()
        if x.strip()
    ]


def looks_like_heading(line):
    """
    Deterministic structural rule for obvious headings.

    The neural model handles ambiguous cases.
    These rules handle obvious standalone headings.
    """

    text = line.strip()

    if not text:
        return False

    # Too long to be a normal heading.
    if len(text) > 100:
        return False

    words = text.split()

    # Very long sentences are not headings.
    if len(words) > 12:
        return False

    # Normal sentences usually end with punctuation.
    if text.endswith((".", "?", "!", ";", ",")):
        return False

    # Explicitly common section names.
    if text.lower() in COMMON_HEADINGS:
        return True

    # ALL CAPS short headings.
    if (
        len(words) <= 8
        and any(c.isalpha() for c in text)
        and text.upper() == text
    ):
        return True

    # Short title-case headings.
    if len(words) <= 8:
        alpha_words = [w for w in words if any(c.isalpha() for c in w)]

        if alpha_words and all(
            w[0].isupper()
            for w in alpha_words
            if w
        ):
            return True

    return False


def build_document(text, title, source, metadata):

    lines = clean_lines(text)

    engine = get_inference_engine()

    sections = []

    current = {
        "heading": title or "Document",
        "level": 1,
        "content": [],
        "items": [],
        "table": None,
    }

    for line in lines:

        # -------------------------------------------------
        # Page markers
        # -------------------------------------------------

        if re.match(r"^\[\[PAGE \d+\]\]$", line):
            continue

        # -------------------------------------------------
        # Explicit Markdown headings
        # -------------------------------------------------

        m = HEADING_RE.match(line)

        if m:

            if (
                current["content"]
                or current["items"]
                or current["heading"] != title
            ):
                sections.append(current)

            current = {
                "heading": m.group(2),
                "level": min(len(m.group(1)), 6),
                "content": [],
                "items": [],
                "table": None,
            }

            continue

        # -------------------------------------------------
        # Lists
        # -------------------------------------------------

        lm = LIST_RE.match(line)

        if lm:
            current["items"].append(lm.group(1))
            continue

        # -------------------------------------------------
        # Obvious structural headings
        #
        # Do this BEFORE neural inference.
        # -------------------------------------------------

        if looks_like_heading(line):

            if current["content"] or current["items"]:
                sections.append(current)

            current = {
                "heading": line,
                "level": 2,
                "content": [],
                "items": [],
                "table": None,
            }

            continue

        # -------------------------------------------------
        # Neural semantic classifier
        # -------------------------------------------------

        kind = engine.classify(line)

        if (
            kind == "heading"
            and len(line) < 100
            and len(line.split()) <= 12
        ):

            if current["content"] or current["items"]:
                sections.append(current)

            current = {
                "heading": line,
                "level": 2,
                "content": [],
                "items": [],
                "table": None,
            }

        else:
            current["content"].append(line)

    # -----------------------------------------------------
    # Final section
    # -----------------------------------------------------

    if (
        current["content"]
        or current["items"]
        or not sections
    ):
        sections.append(current)

    return {
        "title": title or "Untitled document",
        "source": source,
        "sections": sections,
        "toc": [
            {
                "heading": section["heading"],
                "level": section["level"],
            }
            for section in sections
        ],
        "metadata": metadata,
    }