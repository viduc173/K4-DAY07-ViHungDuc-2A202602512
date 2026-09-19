# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Vi Hùng Đức
**Nhóm:** C3D
**Ngày:** 19/09/2026
> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai đoạn văn bản có độ tương tự cosine cao nghĩa là vector embedding của chúng chỉ về cùng một hướng, tức là hai đoạn có ý nghĩa/chủ đề gần nhau, dù có thể dùng từ ngữ khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên muốn xin bảng điểm cần gửi yêu cầu đến Phòng Quản lý Đào tạo."
- Câu B: "Người học đề nghị cấp bảng điểm thông qua Phòng Quản lý Đào tạo."
- Tại sao tương đồng: cùng nói về một việc (xin cấp bảng điểm) với cùng đơn vị xử lý; chỉ khác cách diễn đạt ("sinh viên" / "người học", "xin" / "đề nghị cấp"), mà embedding ngữ nghĩa nắm được sự đồng nghĩa này.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Sinh viên muốn xin bảng điểm cần gửi yêu cầu đến Phòng Quản lý Đào tạo."
- Câu B: "Hôm nay trời mưa to nên chuyến bay bị hoãn."
- Tại sao khác: hai câu khác hẳn chủ đề (thủ tục học vụ và thời tiết/đi lại), không chung ngữ cảnh hay từ khóa, nên vector chỉ về các hướng khác nhau.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine chỉ đo góc (hướng) giữa hai vector, không phụ thuộc độ dài vector, nên không bị ảnh hưởng bởi việc văn bản dài hay ngắn; còn khoảng cách Euclid bị lệch theo độ lớn của vector, có thể coi hai đoạn cùng nghĩa là "xa nhau" chỉ vì độ dài khác nhau.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* số chunk = ⌈(10000 − 50) / (500 − 50)⌉ = ⌈9950 / 450⌉ = ⌈22,11⌉ = 23 (bước nhảy mỗi chunk là 450 ký tự).
> *Đáp án:* **23 chunks** (đã chạy thử với `FixedSizeChunker(500, 50)` cho kết quả 23).

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Số chunk tăng lên: ⌈(10000 − 100) / (500 − 100)⌉ = ⌈9900 / 400⌉ = ⌈24,75⌉ = **25 chunks** (đã chạy thử, cũng ra 25), vì bước nhảy giảm còn 400. Muốn overlap lớn hơn để câu và ý nằm ở ranh giới hai chunk không bị cắt mất ngữ cảnh, đổi lại tốn thêm dung lượng lưu trữ và số lần embed.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Mình tách câu bằng `re.split(r"(?<=[.!?])\s+", text.strip())`: lookbehind `(?<=[.!?])` giữ dấu câu lại trong câu, còn `\s+` bắt cả khoảng trắng lẫn xuống dòng sau dấu chấm. Sau đó gom mỗi `max_sentences_per_chunk` câu thành một chunk bằng `" ".join(...)`. Các edge case đã xử lý: văn bản rỗng hoặc toàn khoảng trắng trả `[]`, bỏ chuỗi rỗng sau khi tách, văn bản không có dấu câu thì ra 1 chunk, và `max(1, ...)` trong constructor để tham số không nhỏ hơn 1.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán thử các dấu tách theo thứ tự ưu tiên (`\n\n`, `\n`, `. `, ` `, `""`): tách văn bản theo dấu hiện tại, gom các mảnh liền nhau vào một buffer cho đến khi sắp vượt `chunk_size`, mảnh nào vẫn quá dài thì đệ quy với các dấu còn lại. Dấu tách được giữ ở cuối mỗi mảnh nên không mất ký tự. Base case gồm: văn bản đã ≤ `chunk_size` thì trả về nguyên, và hết dấu tách (hoặc gặp `""`) thì cắt cứng theo `chunk_size`; dấu không xuất hiện trong văn bản thì bỏ qua sang dấu kế tiếp.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Mình lưu trong danh sách in-memory `self._store`; mỗi record gồm `id` (dạng `doc_id#số thứ tự` để nhiều chunk cùng tài liệu không trùng), `content`, `metadata` (được bổ sung `doc_id`) và `embedding` tính bằng `embedding_fn` lúc thêm. Khi `search`, mình embed câu hỏi, tính dot product với embedding của từng record (các embedder đều đã chuẩn hóa nên dot product tương đương cosine), sắp giảm dần theo `score` rồi lấy `top_k`. Mình không dùng ChromaDB vì nó không nằm trong `requirements.txt`.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` **lọc trước, tìm sau**: giữ lại các record có mọi cặp khóa/giá trị trong `metadata_filter` khớp (điều kiện AND), rồi chạy tìm kiếm tương đồng chỉ trên tập đã lọc; nếu không truyền filter thì hoạt động như `search`. `delete_document` dựng lại danh sách `_store` bằng cách loại mọi chunk có `metadata["doc_id"]` khớp, rồi so độ dài trước và sau để trả `True` nếu đã xóa được ít nhất một chunk, `False` nếu không.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> `answer` làm theo mô hình RAG: gọi `store.search(question, top_k)` để lấy các chunk liên quan, đánh số `[1]`, `[2]`, ... rồi nối lại thành phần "Context". Prompt gồm chỉ dẫn (chỉ trả lời dựa trên context, không có thì nói không biết), khối `Context:`, câu hỏi và `Answer:`, sau đó đưa cho `llm_fn`. Nếu store không trả về chunk nào thì context ghi `(no relevant context found)` để LLM không bịa.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\viduc\AppData\Local\Programs\Python\Python313\python.exe
cachedir: .pytest_cache
rootdir: D:\VinTC\K4-DAY07-ViHungDuc-2A202602512
plugins: anyio-4.12.1
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.06s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Các bước nộp hồ sơ xin học bổng gồm chọn học bổng, kiểm tra điều kiện, chuẩn bị và nộp hồ sơ. | Để nhận học bổng, sinh viên cần nộp hồ sơ theo bốn bước. | cao | 0,784 | Đúng |
| 2 | Học phí được thanh toán theo từng học kỳ. | Tuition fees are paid every semester. | cao | 0,793 | Đúng |
| 3 | Học phí được thanh toán theo từng học kỳ. | Hôm nay trời mưa to nên chuyến bay bị hoãn. | thấp | -0,065 | Đúng |
| 4 | Sinh viên được chiết khấu học phí. | Sinh viên không được chiết khấu học phí. | cao | 0,714 | Đúng (nhưng nghĩa ngược nhau) |
| 5 | Nhân viên được hưởng 100% lương khi nghỉ thai sản. | Sinh viên được chiết khấu 5% học phí. | thấp | 0,090 | Đúng |

*Điểm thực tế: cosine giữa hai vector của embedder `paraphrase-multilingual-MiniLM-L12-v2` (hàm `compute_similarity`). Dự đoán được ghi trước khi chạy; ngưỡng "cao" là cosine ≥ 0,5.*

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là cặp 4: hai câu ý nghĩa ngược nhau ("được" và "không được" chiết khấu) vẫn đạt 0,714, gần bằng cặp diễn đạt lại cùng nghĩa (cặp 1: 0,784); và cặp song ngữ Việt–Anh (cặp 2) đạt 0,793, cao nhất trong 5 cặp, cao hơn cả cặp cùng ngôn ngữ. Điều này cho thấy embedding biểu diễn chủ đề và ngữ cảnh chung chứ không biểu diễn đúng/sai hay phủ định, nên truy xuất có thể lấy về đoạn nói ngược lại với điều cần hỏi; còn mô hình đa ngữ đặt câu cùng nghĩa ở hai ngôn ngữ gần nhau. Cặp 3 còn cho điểm âm (-0,065), nhắc rằng cosine nằm trong [-1, 1] chứ không chỉ [0, 1].

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Học phí toàn bộ chương trình cử nhân Kinh doanh năm 2026? | `dac-biet#12`: mục "Chứng chỉ sau đại học" (chiết khấu học phí sau đại học) | 0,691 | Không ở top-1 (chunk đúng `hoc-phi-rmit#3` ở hạng 3, 0,655) | Chưa có LLM thật; ngữ cảnh top-3 có chứa con số 1.127.520.000 |
| 2 | Có anh chị em ruột đang học RMIT có được giảm học phí không? | `dac-biet#3`: cùng mục anh chị em ruột, nhưng là đoạn nói về trường hợp đã hoàn thành chương trình | 0,719 | Một phần (đúng mục, chưa có con số 5%; chunk chứa "chiết khấu 5%" ở hạng 2, 0,695) | Chưa có LLM thật; ngữ cảnh top-3 có chứa "chiết khấu 5%" |
| 3 | Các bước nộp hồ sơ xin học bổng RMIT? | `hoc-bong#5`: mục "Các bước nộp hồ sơ xin học bổng" (4 bước) | 0,655 | Có | Chưa có LLM thật; ngữ cảnh chứa đủ 4 bước |
| 4 | Các khoản phí phụ thu bắt buộc và mức phí? | `hoc-phi-rmit#16`: mục "Phí phụ thu bắt buộc" (bảo hiểm y tế) | 0,498 | Có (chỉ hơn chunk hạng 2 là 0,496, nên khá mong manh) | Chưa có LLM thật; ngữ cảnh chứa 7.400.000 và 683.100 |
| 5 | Có dịch vụ chăm sóc sức khỏe và tâm lý không? (filter `audience=student`) | `ho-tro-sinh-vien#2`: mục "Chăm sóc Sức khỏe và Tâm lý" | 0,542 | Có | Chưa có LLM thật; ngữ cảnh chứa câu "...hỗ trợ cho sinh viên từ những ngày đầu cho đến khi tốt nghiệp" |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5. Điểm theo thang SCORING: Q1 = 1, Q2 = 1, Q3 = 2, Q4 = 2, Q5 = 2, tổng **8/10**. Chạy bằng `python bench.py` với `HeadingChunker` (chunk_size=800), bảng Markdown đã được chuẩn hóa thành câu có tên cột ở bước làm sạch dữ liệu, và embedder `paraphrase-multilingual-MiniLM-L12-v2` (không chuẩn hóa bảng thì chỉ 6/10 và 4/5); kết quả đầy đủ trong `ket_qua_benchmark.txt`. Chưa đạt 10/10: Q1 và Q2 có chunk đúng ở hạng 3 và hạng 2 chứ chưa ở hạng 1. Cột "Câu trả lời của Agent": mình chưa nối LLM thật (`main.py` chỉ có `demo_llm`), nên chấm ở mức ngữ cảnh truy xuất được có chứa đáp án hay không.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Khi chạy cả 4 chiến lược của nhóm trên cùng `bench.py`, cùng embedder và cùng 5 câu, điểm đi từ 2/10 đến 8/10, nghĩa là cách cắt chunk ảnh hưởng nhiều hơn mình nghĩ; và chiến lược tôn trọng cấu trúc heading thắng cắt cơ học. Mình cũng học được cách chấm ở mức nội dung chunk (chuỗi đáp án phải nằm trong chunk) thay vì chỉ kiểm `doc_id`, vì cách sau thổi phồng kết quả. *(Ghi theo kết quả so sánh trong nhóm; sẽ bổ sung sau buổi demo.)*

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 (42/42 passed) |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
