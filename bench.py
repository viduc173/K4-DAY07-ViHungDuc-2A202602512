"""Chạy 5 câu hỏi benchmark trên corpus data/rmit và in top-3 để đối chiếu gold answer.

Cách dùng:
    python bench.py                                  # HeadingChunker (chiến lược của Đức), embedder theo .env
    python bench.py --strategy recursive
    python bench.py --strategy sentence --sentences 2
    python bench.py --strategy fixed --chunk-size 300
    python bench.py --no-filter                      # bỏ metadata_filter để chạy A/B
    python bench.py --raw-tables                     # không chuẩn hóa bảng thành câu (A/B)
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    LocalEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

CORPUS_DIR = Path("data/rmit")

# Mỗi thành viên chỉ đổi DÒNG NÀY để chọn chunker của mình: heading | recursive | sentence | fixed.
# Mọi thứ khác giữ nguyên để so sánh công bằng. (--strategy trên dòng lệnh chỉ để chạy so sánh nhanh.)
MY_STRATEGY = "heading"

# Sửa cho khớp 5 câu hỏi nhóm đã thống nhất trong REPORT_NHOM.md.
# filter=None nghĩa là tìm trên toàn corpus; ít nhất 1 câu phải có audience.
# "gold" trích từ tài liệu trong data/rmit, chỉ để đối chiếu với top-3 in ra.
# "expect": mọi chuỗi này phải cùng xuất hiện trong MỘT chunk thì mới tính là chunk chứa đáp án
# (chấm ở mức nội dung, không chỉ kiểm doc_id).
QUERIES: list[dict] = [
    {
        "query": "Học phí toàn bộ chương trình cử nhân Kinh doanh năm 2026 là bao nhiêu?",
        "filter": {"category": "tuition"},
        "gold": "1.127.520.000 VND (288 tín chỉ, 24 môn) — chinh-sach-hoc-phi-rmit, Khoa Kinh doanh",
        "expect": ["Kinh doanh", "Toàn chương trình", "1.127.520.000"],
    },
    {
        "query": "Sinh viên có anh chị em ruột đang học tại RMIT có được giảm học phí không?",
        "filter": None,
        "gold": "Có: được chiết khấu 5% học phí khi bắt đầu nhập học trong năm 2026 (không kết hợp được với "
                "các chính sách học phí đặc biệt khác) — chinh-sach-hoc-phi-dac-biet",
        "expect": ["chiết khấu 5%"],
    },
    {
        "query": "Các bước nộp hồ sơ xin học bổng RMIT là gì?",
        "filter": {"category": "scholarship"},
        "gold": "4 bước: chọn học bổng phù hợp, kiểm tra điều kiện, chuẩn bị hồ sơ, nộp hồ sơ — hoc-bong-rmit-vietnam",
        "expect": ["Chọn học bổng phù hợp", "Kiểm tra điều kiện nộp học bổng", "Chuẩn bị hồ sơ", "Nộp hồ sơ"],
    },
    {
        "query": "Những khoản phí phụ thu bắt buộc gồm những gì và mức phí bao nhiêu?",
        "filter": {"category": "tuition"},
        "gold": "Bảo hiểm y tế dành cho sinh viên quốc tế: 7.400.000 VND/học kỳ; Bảo hiểm y tế bắt buộc: "
                "683.100 VND/năm, áp dụng từ học kỳ 2 năm 2026 — chinh-sach-hoc-phi-rmit, mục Phí phụ thu bắt buộc",
        "expect": ["Bảo hiểm y tế bắt buộc", "7.400.000", "683.100"],
    },
    {
        # Câu CẦN filter: không nêu người hỏi là ai, và cả tài liệu sinh viên lẫn nhân viên đều có
        # mục về sức khỏe/y tế. Không lọc thì tài liệu nhân viên chiếm top-1.
        "query": "Có dịch vụ chăm sóc sức khỏe và tâm lý không?",
        "filter": {"audience": "student"},
        "gold": "Mục Chăm sóc Sức khỏe và Tâm lý: \"Dịch vụ hỗ trợ và mạng lưới của chúng tôi có thể hỗ trợ cho "
                "sinh viên từ những ngày đầu cho đến khi tốt nghiệp.\" — ho-tro-sinh-vien-rmit",
        "expect": ["Chăm sóc Sức khỏe và Tâm lý", "sinh viên từ những ngày đầu"],
    },
]


class HeadingChunker:
    """Tách trước mỗi dòng heading Markdown; mỗi section là một chunk.

    Section dài hơn chunk_size thì cắt theo dòng, và gắn lại tiêu đề (cùng dòng tiêu đề cột nếu là
    bảng) vào từng mảnh con để mảnh nào cũng còn ngữ cảnh "đây là mục nói về cái gì".
    """

    def __init__(self, chunk_size: int = 800) -> None:
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        chunks: list[str] = []
        for section in re.split(r"(?m)^(?=#{1,6}\s)", text):
            section = section.strip()
            if not section:
                continue
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                chunks.extend(self._split_section(section))
        return chunks

    @staticmethod
    def _table_header(lines: list[str]) -> str:
        for i in range(len(lines) - 1):
            if lines[i].startswith("|") and re.fullmatch(r"\|[\s:|-]+\|", lines[i + 1].strip()):
                return lines[i] + "\n" + lines[i + 1]
        return ""

    def _split_section(self, section: str) -> list[str]:
        lines = section.split("\n")
        heading = lines[0] if lines[0].startswith("#") else ""
        body = lines[1:] if heading else lines
        table_header = self._table_header(body)
        header_lines = set(table_header.split("\n")) if table_header else set()

        def prefix_for(line: str) -> str:
            prefix = heading
            if table_header and line.startswith("|"):
                prefix += ("\n" if prefix else "") + table_header
            return prefix

        def join(prefix: str, current: str) -> str:
            return prefix + "\n" + current if prefix else current

        pieces: list[str] = []
        prefix, current = "", ""

        def flush() -> None:
            nonlocal current
            if current:
                pieces.append(join(prefix, current))
                current = ""

        for line in body:
            if not line.strip() or line in header_lines:
                continue
            room = self.chunk_size - len(prefix_for(line)) - 1
            if len(line) > room:
                # Một dòng quá dài: hạ xuống RecursiveChunker, mỗi mảnh vẫn gắn lại tiêu đề.
                flush()
                prefix = prefix_for(line)
                pieces.extend(join(prefix, part) for part in RecursiveChunker(chunk_size=max(room, 50)).chunk(line))
                continue
            if not current:
                prefix = prefix_for(line)
            candidate = current + ("\n" if current else "") + line
            if current and len(prefix) + 1 + len(candidate) > self.chunk_size:
                flush()
                prefix, current = prefix_for(line), line
            else:
                current = candidate
        flush()
        return pieces


def parse_markdown(path: Path) -> tuple[dict[str, str], str]:
    """Tách frontmatter YAML đơn giản (key: "value") và phần thân."""
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text.strip()
    pairs = re.findall(r"^(\w+):\s*(.+)$", parts[1], re.M)
    metadata = {key: value.strip().strip('"') for key, value in pairs}
    return metadata, parts[2].strip()


def build_chunker(args: argparse.Namespace):
    if args.strategy == "heading":
        return HeadingChunker(chunk_size=args.chunk_size or 800)
    args.chunk_size = args.chunk_size or 300
    if args.strategy == "fixed":
        return FixedSizeChunker(chunk_size=args.chunk_size, overlap=args.overlap)
    if args.strategy == "sentence":
        return SentenceChunker(max_sentences_per_chunk=args.sentences)
    return RecursiveChunker(chunk_size=args.chunk_size)


def normalize_tables(text: str) -> str:
    """Bước làm sạch dữ liệu: đổi mỗi dòng bảng Markdown thành một câu có tên cột.

    Embedding xếp bảng toàn số thấp hơn văn xuôi, nên "| Kinh doanh | Toàn chương trình | 288 |..." được đổi thành
    "Chương trình: Kinh doanh, Phạm vi: Toàn chương trình, Tín chỉ: 288, ...". Áp dụng chung cho mọi chiến lược.
    """
    out: list[str] = []
    header: list[str] | None = None
    for line in text.split("\n"):
        if not line.startswith("|"):
            header = None
            out.append(line)
            continue
        if re.fullmatch(r"\|[\s:|-]+\|", line.strip()):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        out.append(", ".join(f"{h}: {c}" for h, c in zip(header, cells)) + ".")
    return "\n".join(out)


def build_documents(chunker, clean_tables: bool = True) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        frontmatter, body = parse_markdown(path)
        if clean_tables:
            body = normalize_tables(body)
        for i, chunk in enumerate(chunker.chunk(body)):
            documents.append(
                Document(
                    id=f"{path.stem}#{i}",
                    content=chunk,
                    metadata={**frontmatter, "doc_id": path.stem, "chunk_index": i},
                )
            )
    return documents


def build_embedder():
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception as error:
            print(f"Không tải được LocalEmbedder ({error}); dùng mock.")
    return _mock_embed


def contains_answer(content: str, expect: list[str]) -> bool:
    lowered = content.lower()
    return all(fragment.lower() in lowered for fragment in expect)


def grade(results: list[dict], expect: list[str]) -> int:
    """2đ: top-1 chứa đáp án; 1đ: top-2/3 chứa đáp án; 0đ: không chunk nào chứa."""
    for rank, result in enumerate(results, start=1):
        if contains_answer(result["content"], expect):
            return 2 if rank == 1 else 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=["heading", "recursive", "sentence", "fixed"], default=MY_STRATEGY)
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Mặc định: 800 cho heading, 300 cho các chiến lược khác")
    parser.add_argument("--overlap", type=int, default=0, help="Chỉ dùng cho fixed")
    parser.add_argument("--sentences", type=int, default=3, help="Chỉ dùng cho sentence")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-filter", action="store_true", help="Bỏ metadata_filter (chạy A/B)")
    parser.add_argument("--raw-tables", action="store_true", help="Không chuẩn hóa bảng thành câu (chạy A/B)")
    args = parser.parse_args()

    embedder = build_embedder()
    documents = build_documents(build_chunker(args), clean_tables=not args.raw_tables)
    store = EmbeddingStore(collection_name="bench", embedding_fn=embedder)
    store.add_documents(documents)

    backend = getattr(embedder, "_backend_name", embedder.__class__.__name__)
    print(f"Chiến lược: {args.strategy} | Embedder: {backend} | Filter: {'TẮT' if args.no_filter else 'BẬT'}")
    print(f"Đã nạp {store.get_collection_size()} chunk từ {len(list(CORPUS_DIR.glob('*.md')))} tài liệu")
    if backend == "mock embeddings fallback":
        print("Lưu ý: mock embedder không có ngữ nghĩa, điểm chỉ để kiểm tra cơ chế. "
              "Đặt EMBEDDING_PROVIDER=local để đánh giá thật.")

    total = 0
    in_top3 = 0
    for number, item in enumerate(QUERIES, start=1):
        metadata_filter = None if args.no_filter else item["filter"]
        print(f"\n=== Q{number}: {item['query']}  (filter={metadata_filter})")
        print(f"    GOLD: {item['gold']}")
        results = store.search_with_filter(item["query"], top_k=args.top_k, metadata_filter=metadata_filter)
        for rank, result in enumerate(results, start=1):
            preview = result["content"][:110].replace("\n", " ")
            mark = "  <-- chứa đáp án" if contains_answer(result["content"], item["expect"]) else ""
            print(f"  {rank}. score={result['score']:.3f} doc_id={result['metadata']['doc_id']}"
                  f"#{result['metadata']['chunk_index']}{mark}")
            print(f"     {preview}...")
        points = grade(results, item["expect"])
        total += points
        in_top3 += points > 0
        print(f"  => điểm Q{number}: {points}/2")

    print(f"\nTỔNG: {total}/{2 * len(QUERIES)} | có chunk chứa đáp án trong top-{args.top_k}: "
          f"{in_top3}/{len(QUERIES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
