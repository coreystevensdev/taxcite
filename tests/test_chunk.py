from taxcite.chunk import TARGET_CHARS, Chunk, chunk_pages
from taxcite.parse import Page


def page(number, *paragraphs):
    return Page(number=number, text="\n\n".join(paragraphs))


class TestChunkPages:
    def test_small_input_yields_single_chunk(self):
        chunks = chunk_pages("p501", [page(1, "First paragraph.", "Second one.")])
        assert len(chunks) == 1
        assert chunks[0] == Chunk(
            pub_id="p501",
            ordinal=0,
            first_page=1,
            last_page=1,
            text="First paragraph.\n\nSecond one.",
        )

    def test_empty_pages_yield_no_chunks(self):
        assert chunk_pages("p501", [page(1, ""), page(2, "   ")]) == []

    def test_splits_when_over_target(self):
        big = "x" * (TARGET_CHARS // 2)
        chunks = chunk_pages("p17", [page(1, big, big, big)])
        assert len(chunks) >= 2

    def test_neighbors_share_one_paragraph_of_overlap(self):
        paras = [f"paragraph {i} " + "y" * 700 for i in range(5)]
        chunks = chunk_pages("p17", [page(1, *paras)])
        assert len(chunks) >= 2
        first_tail = chunks[0].text.split("\n\n")[-1]
        second_head = chunks[1].text.split("\n\n")[0]
        assert first_tail == second_head

    def test_page_range_spans_source_pages(self):
        big = "z" * 900
        chunks = chunk_pages("p936", [page(3, big), page(4, big), page(5, big)])
        assert chunks[0].first_page == 3
        assert chunks[-1].last_page == 5
        for chunk in chunks:
            assert chunk.first_page <= chunk.last_page

    def test_ordinals_are_sequential(self):
        big = "w" * 900
        chunks = chunk_pages("p17", [page(1, big, big, big, big)])
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    def test_pdf_style_page_without_blank_lines_still_hits_target(self):
        # extract_text output: hundreds of short lines, no blank-line breaks
        lines = "\n".join(f"line {i} of dense two-column body text." for i in range(400))
        chunks = chunk_pages("p501", [page(1, lines)])
        assert len(chunks) > 1
        assert all(len(c.text) <= TARGET_CHARS * 1.5 for c in chunks)

    def test_paragraph_larger_than_target_becomes_own_chunk(self):
        huge = "q" * (TARGET_CHARS * 2)
        chunks = chunk_pages("p17", [page(1, "small intro.", huge, "small outro.")])
        assert any(huge in c.text for c in chunks)


class TestSentenceBoundaries:
    """Chunks used to start and end mid-word because oversized blocks were packed
    on PDF line breaks, which fall wherever the text happened to wrap."""

    def test_reflow_rejoins_a_word_broken_across_lines(self):
        from taxcite.chunk import _reflow

        assert _reflow("the standard de-\nduction is") == "the standard deduction is"

    def test_reflow_joins_wrapped_lines_into_flowing_text(self):
        from taxcite.chunk import _reflow

        assert _reflow("you can claim\nthe credit") == "you can claim the credit"

    def test_does_not_split_on_common_irs_abbreviations(self):
        from taxcite.chunk import _pack_sentences

        block = ("See Pub. 501 for the rules. " * 20) + ("It applies to U.S. filers only. " * 20)
        for piece in _pack_sentences(block):
            assert not piece.startswith("501")
            assert not piece.startswith("filers")

    def test_pieces_begin_at_a_sentence_not_mid_word(self):
        from taxcite.chunk import _pack_sentences

        block = " ".join(
            f"This is sentence number {i} about deductions and credits." for i in range(40)
        )
        pieces = _pack_sentences(block)
        assert len(pieces) > 1
        assert all(p[0].isupper() for p in pieces)

    def test_a_block_with_no_sentences_still_gets_split(self):
        """Tables and index runs have no sentence boundaries. Without a backstop the
        whole block came back as one 15,889-character chunk."""
        from taxcite.chunk import TARGET_CHARS, _pack_sentences

        block = " ".join(f"line {i} of dense body text." for i in range(400))
        pieces = _pack_sentences(block)
        assert len(pieces) > 1
        assert all(len(p) <= TARGET_CHARS for p in pieces)
