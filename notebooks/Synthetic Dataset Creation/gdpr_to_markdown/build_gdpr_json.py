import os
import re
import json
from pathlib import Path


def parse_article(file_path):
    """Parse an Article markdown file and extract structured data."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract article number from filename
    filename = os.path.basename(file_path)
    article_number = int(re.search(r"Article (\d+)", filename).group(1))

    # Split content into main text and suitable recitals section
    parts = content.split("## Suitable Recitals")
    main_text = parts[0].strip()
    recitals_section = parts[1].strip() if len(parts) > 1 else ""

    # Extract related recitals
    related_recitals = []
    if recitals_section:
        recital_matches = re.findall(r"\((\d+)\)", recitals_section)
        related_recitals = [int(num) for num in recital_matches]

    # Parse paragraphs from main text
    paragraphs = parse_paragraphs(main_text)

    # Generate full text (concatenate all paragraph texts)
    full_text = extract_full_text(paragraphs)

    # Get article title (if present in content or generate default)
    title = f"Article {article_number}"
    # Try to extract title from first line if it looks like a heading
    first_line = main_text.split("\n")[0] if main_text else ""
    if first_line.startswith("#"):
        # Remove markdown heading syntax
        title = re.sub(r"^#+\s*", "", first_line).strip()

    article = {
        "title": title,
        "number": article_number,
        "fullText": full_text,
        "relatedRecitals": related_recitals,
        "paragraphs": paragraphs,
    }

    return article


def parse_paragraphs(text):
    """Parse numbered paragraphs and nested subparagraphs from markdown text."""
    paragraphs = []

    # Split by numbered items at the root level (1., 2., etc.)
    lines = text.split("\n")
    current_paragraph = None
    current_text = []

    for line in lines:
        # Check for root-level numbered paragraph (1., 2., etc.)
        root_match = re.match(r"^(\d+)\.\s+(.+)$", line)
        if root_match:
            # Save previous paragraph if exists
            if current_paragraph is not None:
                para_text = "\n".join(current_text).strip()
                current_paragraph["text"] = para_text
                subparagraphs = parse_subparagraphs(para_text)
                current_paragraph["subparagraphs"] = subparagraphs
                paragraphs.append(current_paragraph)

            # Start new paragraph
            para_content = root_match.group(2)
            current_paragraph = {
                "text": para_content,
                "relatedRecitals": [],
                "subparagraphs": [],
            }
            current_text = [line]
        elif current_paragraph is not None:
            current_text.append(line)

    # Add last paragraph
    if current_paragraph is not None:
        para_text = "\n".join(current_text).strip()
        current_paragraph["text"] = para_text
        subparagraphs = parse_subparagraphs(para_text)
        current_paragraph["subparagraphs"] = subparagraphs
        paragraphs.append(current_paragraph)

    # If no numbered paragraphs found, treat entire text as single paragraph
    if not paragraphs and text.strip():
        clean_text = clean_paragraph_text(text.strip())
        paragraphs.append(
            {"text": clean_text, "relatedRecitals": [], "subparagraphs": []}
        )

    return paragraphs


def parse_subparagraphs(text):
    """Parse nested subparagraphs (a), b), i), ii), etc.) from paragraph text."""
    subparagraphs = []

    # Match different levels of subparagraphs
    # Level 1: 1., 2., 3. (indented with spaces/tabs)
    # Level 2: a), b), c) or (a), (b), (c)
    # Level 3: i), ii), iii) or (i), (ii), (iii)

    lines = text.split("\n")
    for line in lines:
        # Check for subparagraph markers (indented numbered or lettered items)
        sub_match = re.match(
            r"^\s+(\d+|[a-z]+|[ivxlcdm]+)[\.\)]\s+(.+)$", line, re.IGNORECASE
        )
        if sub_match:
            sub_text = clean_paragraph_text(line.strip())
            subparagraphs.append(
                {"text": sub_text, "relatedRecitals": [], "subparagraphs": []}
            )

    return subparagraphs


def clean_paragraph_text(text):
    """Clean paragraph text by removing markdown formatting and extra whitespace."""
    # Remove markdown links but keep the text
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def extract_full_text(paragraphs):
    """Extract full text from paragraphs structure."""
    full_text_parts = []

    for para in paragraphs:
        if para.get("text"):
            full_text_parts.append(para["text"])

    return " ".join(full_text_parts)


def parse_recital(file_path):
    """Parse a Recital markdown file and extract structured data."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract recital number from filename
    filename = os.path.basename(file_path)
    recital_number = int(re.search(r"Recital (\d+)", filename).group(1))

    # Remove navigation links at the bottom
    content = re.sub(r"\[←Recital \d+\].*$", "", content, flags=re.DOTALL)
    content = re.sub(r"\[Recital \d+→\].*$", "", content, flags=re.DOTALL)

    # Remove "* This title is an unofficial description." text
    content = re.sub(r"\\\*\s*This title is an unofficial description\.", "", content)

    # Clean up the text
    text = clean_paragraph_text(content.strip())

    # Extract sentences (numbered with superscript numbers like 1, 2, 3)
    # In markdown, these appear as plain numbers at the start
    sentences = []
    sentence_matches = re.findall(r"(\d+)([A-Z][^\.]+\.)", text)

    if sentence_matches:
        for _, sentence in sentence_matches:
            sentences.append(sentence.strip())
    else:
        # If no sentence numbers found, treat as single sentence
        sentences = [text]

    recital = {"number": recital_number, "text": text, "sentences": sentences}

    return recital


def build_gdpr_json():
    """Build the complete GDPR JSON structure from markdown files."""
    articles_dir = Path(__file__).parent / "Articles"

    # Parse all articles
    articles = []
    article_files = sorted(
        [f for f in articles_dir.glob("Article *.md")],
        key=lambda x: int(re.search(r"Article (\d+)", x.name).group(1)),
    )

    print(f"Parsing {len(article_files)} articles...")
    for article_file in article_files:
        try:
            article = parse_article(article_file)
            articles.append(article)
            print(f"  ✓ Parsed {article_file.name}")
        except Exception as e:
            print(f"  ✗ Error parsing {article_file.name}: {e}")

    # Parse all recitals
    recitals = []
    recital_files = sorted(
        [f for f in articles_dir.glob("Recital *.md")],
        key=lambda x: int(re.search(r"Recital (\d+)", x.name).group(1)),
    )

    print(f"\nParsing {len(recital_files)} recitals...")
    for recital_file in recital_files:
        try:
            recital = parse_recital(recital_file)
            recitals.append(recital)
            print(f"  ✓ Parsed {recital_file.name}")
        except Exception as e:
            print(f"  ✗ Error parsing {recital_file.name}: {e}")

    # Build final JSON structure
    gdpr_json = {"articles": articles, "recitals": recitals}

    # Save to file
    output_file = Path(__file__).parent / "gdpr.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(gdpr_json, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Successfully created {output_file}")
    print(f"  Articles: {len(articles)}")
    print(f"  Recitals: {len(recitals)}")


if __name__ == "__main__":
    build_gdpr_json()
