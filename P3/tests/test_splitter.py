from src.ingestion.splitter import TextSplitter
from src.schemas.documents import DocumentRecord


def test_splitter_preserves_offsets_and_overlap() -> None:
    text = " ".join(f"token{i}" for i in range(30))
    document = DocumentRecord(
        doc_id="doc-1",
        dataset="toy",
        source_name="Source A",
        full_text=text,
    )

    splitter = TextSplitter(chunk_size=10, chunk_overlap=2)
    chunks = splitter.split_document(document)

    assert len(chunks) == 4
    assert chunks[0].chunk_index == 0
    assert chunks[0].char_start == 0
    assert chunks[0].char_end is not None
    assert "token8 token9" in chunks[1].text
    assert chunks[-1].doc_id == "doc-1"


def test_sentence_aware_chunking_keeps_sentence_boundaries() -> None:
    text = (
        "Alpha released the report on Monday. "
        "The report confirmed the merger details. "
        "Beta denied the timeline in public remarks. "
        "A regulator later requested new filings."
    )
    document = DocumentRecord(
        doc_id="doc-2",
        dataset="toy",
        source_name="Source B",
        title="Alpha Report",
        full_text=text,
    )

    splitter = TextSplitter(
        chunk_size=10,
        chunk_overlap=2,
        chunking_version="v2",
        sentence_aware_chunking=True,
        min_chunk_tokens=4,
        target_chunk_tokens=12,
        max_chunk_tokens=20,
        sentence_overlap=1,
        min_sentences_per_chunk=2,
    )
    chunks = splitter.split_document(document)

    assert chunks
    assert all(chunk.metadata["chunking_version"] == "v2" for chunk in chunks)
    assert all(chunk.text[0].isalnum() for chunk in chunks)
    assert len(chunks) == 2
    assert chunks[0].metadata["sentence_start_index"] == 0
    assert chunks[0].metadata["sentence_end_index"] == 1
    assert chunks[1].metadata["sentence_start_index"] == 1
    assert chunks[1].metadata["sentence_end_index"] == 3
    assert any("The report confirmed the merger details." in chunk.text for chunk in chunks)
    assert "Alpha released the report on Monday." in chunks[0].text
    assert "Beta denied the timeline in public remarks." in chunks[1].text
    assert "A regulator later requested new filings." in chunks[1].text


def test_sentence_aware_chunking_merges_short_trailing_fragments() -> None:
    text = (
        "Gamma published a detailed review of the policy change. "
        "Officials disputed the funding numbers in later testimony. "
        "Meanwhile."
    )
    document = DocumentRecord(
        doc_id="doc-3",
        dataset="toy",
        source_name="Source C",
        title="Gamma Review",
        full_text=text,
    )

    splitter = TextSplitter(
        chunk_size=10,
        chunk_overlap=2,
        chunking_version="v2",
        sentence_aware_chunking=True,
        min_chunk_tokens=4,
        target_chunk_tokens=14,
        max_chunk_tokens=24,
        sentence_overlap=0,
        min_sentences_per_chunk=2,
    )
    chunks = splitter.split_document(document)

    assert len(chunks) == 1
    assert "Meanwhile." in chunks[0].text
    assert chunks[0].metadata["sentence_start_index"] == 0
    assert chunks[0].metadata["sentence_end_index"] == 2
