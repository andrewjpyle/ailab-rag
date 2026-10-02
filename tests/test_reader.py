from ailab_rag.providers import StubProvider
from ailab_rag.reader import REFUSAL, Reader, lexical_overlap


def _supporting() -> list[tuple[str, str]]:
    return [
        ("d#1", "the relay uploads changes every ninety seconds by default on slow links"),
        ("d#0", "nimbus is a backup and sync service"),
    ]


def test_lexical_overlap_fraction() -> None:
    assert lexical_overlap("alpha beta", "alpha gamma") == 0.5
    assert lexical_overlap("", "anything") == 0.0


def test_reader_answers_when_supported_and_cites() -> None:
    reader = Reader(StubProvider(), support_threshold=0.3)
    result = reader.answer("how often does the relay upload changes", _supporting())
    assert not result.refused
    assert result.citations == ("d#1",)
    assert "ninety seconds" in result.answer


def test_reader_refuses_when_unsupported() -> None:
    reader = Reader(StubProvider(), support_threshold=0.9)
    result = reader.answer("what is the enterprise museum pricing tier", _supporting())
    assert result.refused
    assert result.answer == REFUSAL
    assert result.citations == ()


def test_force_answer_skips_refusal() -> None:
    reader = Reader(StubProvider(), support_threshold=0.99, force_answer=True)
    result = reader.answer("totally unrelated query zzz", _supporting())
    assert not result.refused
    assert result.citations  # it cited something despite weak support


def test_sabotage_citations_picks_an_unsupporting_chunk() -> None:
    reader = Reader(StubProvider(), support_threshold=0.0, sabotage_citations=True)
    retrieved = [
        ("d#1", "the relay uploads changes every ninety seconds by default"),
        ("d#2", "an entirely different topic about invoices and refunds"),
    ]
    result = reader.answer("how often does the relay upload changes", retrieved)
    # The answer is copied from d#1, so the sabotaged citation points at d#2.
    assert result.citations == ("d#2",)


def test_parse_handles_multiple_citations() -> None:
    answer, cites = Reader.parse("ANSWER: hi there\nCITE: a, b , c")
    assert answer == "hi there"
    assert cites == ["a", "b", "c"]


def test_reader_refuses_on_no_retrieved_chunks() -> None:
    reader = Reader(StubProvider(), support_threshold=0.3)
    assert reader.answer("anything", []).refused
