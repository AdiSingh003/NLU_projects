"""
Task 1: Scrapes textual data from 3+ IIT Jodhpur web sources.

Sources:
  1. Academic Regulations         → https://iitj.ac.in/office-of-academics/en/academic-regulations
  2. Office of Academics Circulars→ https://iitj.ac.in/office-of-academics/en/circulars
  3. CSE Department Projects      → https://www.iitj.ac.in/computer-science-engineering/en/projects

Output  : raw_corpus.txt
"""

import os
import re
import logging
import requests
import urllib3
from bs4 import BeautifulSoup

# ── Silence SSL warnings (IITJ uses self-signed certs in some places) ─────────
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/collect_data.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Source URLs ───────────────────────────────────────────────────────────────
SOURCES = [
    {
        "name": "Academic Regulations",
        "url":  "https://iitj.ac.in/office-of-academics/en/academic-regulations",
    },
    {
        "name": "Office of Academics – Circulars",
        "url":  "https://iitj.ac.in/office-of-academics/en/circulars",
    },
    {
        "name": "CSE Department – Projects",
        "url":  "https://www.iitj.ac.in/computer-science-engineering/en/projects",
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

# Tags that hold boilerplate — stripped before text extraction
NOISE_TAGS = ["script", "style", "nav", "footer", "header", "noscript", "iframe"]


def fetch_page(url: str, name: str) -> str | None:
    """Download a URL and return the raw HTML, or None on failure."""
    try:
        log.info(f"Fetching [{name}]: {url}")
        resp = requests.get(url, headers=HEADERS, timeout=30, verify=False)
        resp.raise_for_status()
        log.info(f"  → HTTP {resp.status_code}  ({len(resp.content):,} bytes)")
        return resp.text
    except Exception as exc:
        log.error(f"  ✗ Failed to fetch '{name}': {exc}")
        return None


def extract_text(html: str) -> str:
    """
    Parse HTML with BeautifulSoup, strip noise tags, and return clean prose text.
    Each visible text block goes on its own line.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove noisy tags entirely
    for tag in soup(NOISE_TAGS):
        tag.decompose()

    # Collect text blocks from meaningful HTML elements
    lines = []
    for elem in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "td", "th", "span", "div"]):
        # Get the element's text (no descendants' HTML)
        text = elem.get_text(separator=" ", strip=True)

        # Keep only ASCII-printable content, collapse whitespace
        text = re.sub(r"[^\x20-\x7E]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Skip very short fragments (navigation links, numbers, etc.)
        if len(text.split()) >= 5:
            lines.append(text)

    return "\n".join(lines)


def collect():
    """Main entry point: fetch all sources and write raw_corpus.txt."""
    all_text = []

    for src in SOURCES:
        html = fetch_page(src["url"], src["name"])
        if html is None:
            log.warning(f"  Skipping source: {src['name']}")
            continue

        text = extract_text(html)
        words_in_source = len(text.split())
        log.info(f"  Extracted {words_in_source:,} words from [{src['name']}]")
        all_text.append(f"# SOURCE: {src['name']}\n{text}")

    if not all_text:
        log.error("No data was collected. Check network access to IITJ servers.")
        return

    corpus = "\n\n".join(all_text)
    total_words = len(corpus.split())

    os.makedirs("p1_new", exist_ok=True)
    out_path = "raw_corpus.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    log.info("=" * 60)
    log.info(f"Total words collected : {total_words:,}")
    log.info(f"Number of sources     : {len(all_text)}")
    log.info(f"Output written to     : {out_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    collect()
