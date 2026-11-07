"""Text extraction from documents, and downloading web pages.

Each format has its own path: PDFs through pymupdf4llm with OCR, or Docling when layout and
tables matter; HTML through our own parser (regular expressions let Wikipedia's JSON
through); and DOCX/PPTX/XLSX always through Docling.

It used to live inside rag.py, which grew to 1,200 lines mixing extraction, chunking, indexing and search.
"""
import io
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from ..config import settings


# ---------------- extraction ----------------
def _pdf_pages(raw: bytes) -> list[tuple[str, dict]]:
    """PDF → Markdown per page (headings, lists and tables) with pymupdf4llm; plain text if that fails."""
    import pymupdf
    doc = pymupdf.open(stream=raw, filetype="pdf")
    try:
        import pymupdf4llm
        pages = pymupdf4llm.to_markdown(doc, page_chunks=True, show_progress=False)
        out = []
        for i, p in enumerate(pages):
            meta = p.get("metadata", {})
            out.append((p.get("text", ""), {"page": int(meta.get("page_number", meta.get("page", i + 1)) or i + 1)}))
        if any(t.strip() for t, _ in out):
            return out
    except Exception as e:  # noqa: BLE001
        print("[rag] pymupdf4llm failed, using plain text:", e)
    return [(page.get_text(), {"page": i + 1}) for i, page in enumerate(doc)]


def _docx(raw: bytes) -> str:
    """DOCX → Markdown: headings as #, tables as | a | b | rows."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(io.BytesIO(raw))
    lines = []
    for block in d.element.body.iterchildren():
        tag = block.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(block, d)
            t = p.text.strip()
            if not t:
                continue
            style = (p.style.name or "").lower() if p.style is not None else ""
            m = re.match(r"(heading|título|titulo)\s*(\d)", style)
            if m:
                lines.append("#" * int(m.group(2)) + " " + t)
            elif style == "title":
                lines.append("# " + t)
            elif "list" in style:
                lines.append("- " + t)
            else:
                lines.append(t)
        elif tag == "tbl":
            for row in Table(block, d).rows:
                lines.append("| " + " | ".join(c.text.strip() for c in row.cells) + " |")
        lines.append("")
    return "\n".join(lines)


class _HTMLToMarkdown(HTMLParser):
    """HTML → simple Markdown with a real parser (robust to attributes with '>' inside)."""
    SKIP = {"script", "style", "nav", "footer", "header", "noscript", "svg", "button", "form"}
    BLOCK = {"p", "div", "section", "article", "tr", "table", "ul", "ol", "blockquote", "figure", "dl"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0  # depth inside ignored elements

    # Void elements: they never have a closing tag, so they mustn't count towards the ignored depth
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr", "param"}

    def handle_starttag(self, tag, attrs):
        if tag in self.VOID:
            if tag == "br" and not self.skip:
                self.out.append("\n")
            return
        a = dict(attrs)
        # Footnotes like [16] and maintenance/navigation boxes
        cls = a.get("class") or ""
        if self.skip or tag in self.SKIP or (tag == "sup" and "reference" in cls) or "navbox" in cls or "mw-editsection" in cls:
            self.skip += 1
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag in ("td", "th"):
            self.out.append(" | ")
        elif tag in self.BLOCK:
            self.out.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if self.skip:
            self.skip -= 1
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.out.append("\n\n")
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_startendtag(self, tag, attrs):
        if not self.skip and tag == "br":
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def _html(text: str) -> str:
    p = _HTMLToMarkdown()
    p.feed(text)
    md = "".join(p.out)
    md = re.sub(r"[ \t\xa0]+", " ", md)
    md = re.sub(r" *\n *", "\n", md)
    # A heading must be on a single line
    md = re.sub(r"^(#{1,6}) *\n+", r"\1 ", md, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", md).strip()


_docling = None
DOCLING_EXT = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md"}


def _docling_converter():
    global _docling
    if _docling is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True)  # OCR: usa el parser "auto" (Tesseract)
        _docling = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return _docling


def _docling_pages(filename: str, raw: bytes, ext: str) -> list[tuple[str, dict]]:
    """Docling: layout analysis (columns, headers, footers) and table structure → Markdown per page."""
    from docling.datamodel.base_models import DocumentStream
    stream = DocumentStream(name=Path(filename).stem + ext, stream=io.BytesIO(raw))
    doc = _docling_converter().convert(stream).document
    pages = sorted(doc.pages.keys()) if getattr(doc, "pages", None) else []
    if len(pages) > 1:
        out = [(doc.export_to_markdown(page_no=p), {"page": p}) for p in pages]
        if any(t.strip() for t, _ in out):
            return out
    return [(doc.export_to_markdown(), {})]


def extract(filename: str, raw: bytes, ext: str | None = None, parser: str = "auto") -> list[tuple[str, dict]]:
    """Returns [(markdown, metadata)] per page/document. `ext` forces the type (e.g. for URLs)."""
    ext = ext or Path(filename).suffix.lower()
    if (parser == "docling" and ext in DOCLING_EXT) or ext in (".pptx", ".xlsx"):
        try:
            return _docling_pages(filename, raw, ext)
        except Exception as e:  # noqa: BLE001
            if ext in (".pptx", ".xlsx"):
                raise
            print(f"[rag] Docling failed with {filename} ({e}); using the fast parser")
    if ext == ".pdf":
        return _pdf_pages(raw)
    if ext == ".docx":
        return [(_docx(raw), {})]
    text = raw.decode("utf-8-sig", errors="replace")
    if ext in (".html", ".htm"):
        text = _html(text)
    return [(text, {})]



# ---------------- web connector ----------------
UA = "Mozilla/5.0 (X11; Linux x86_64) LLMTrainer/0.1"


def fetch_url(url: str, timeout: int = 20) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,application/pdf,*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(15_000_000)


def sitemap_urls(url: str, limit: int) -> list[str]:
    xml = fetch_url(url).decode("utf-8", "replace")
    locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml)
    out: list[str] = []
    for loc in locs:
        if loc.endswith(".xml") and len(out) < limit:  # nested sitemap index
            try:
                out += sitemap_urls(loc, limit - len(out))
            except Exception:  # noqa: BLE001
                pass
        else:
            out.append(loc)
        if len(out) >= limit:
            break
    return out[:limit]


def url_ext(url: str, raw: bytes) -> str:
    """Document type of a downloaded URL, by content (the URL may have no extension)."""
    if raw[:5] == b"%PDF-":
        return ".pdf"
    suffix = Path(url.split("?")[0]).suffix.lower()
    return suffix if suffix in (".md", ".txt", ".docx") else ".html"


