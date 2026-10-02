from ailab_rag.gen_checks import citation_resolvable, classify_refusal, lexical_support


def test_citation_resolvable() -> None:
    assert citation_resolvable(["a", "b"], ["a", "b", "c"]) == 1.0
    assert citation_resolvable(["a", "z"], ["a", "b"]) == 0.5
    assert citation_resolvable([], ["a"]) == 0.0  # an uncited claim does not resolve


def test_lexical_support_high_when_answer_is_in_chunk() -> None:
    answer = "uploads every ninety seconds by default"
    chunk = "the relay uploads every ninety seconds by default on slow links"
    assert lexical_support(answer, chunk) > 0.8


def test_lexical_support_negative_control_scores_near_zero() -> None:
    # Citing an unsupporting chunk: the answer n-grams are not in it.
    answer = "uploads every ninety seconds by default"
    unsupporting = "invoices refunds billing receipts and payment plans"
    assert lexical_support(answer, unsupporting) < 0.2


def test_lexical_support_empty_answer_is_zero() -> None:
    assert lexical_support("", "anything here") == 0.0


def test_lexical_support_unigram_fallback_for_one_word_answer() -> None:
    assert lexical_support("ninety", "uploads ninety seconds") == 1.0
    assert lexical_support("missing", "uploads ninety seconds") == 0.0


def test_classify_refusal_four_buckets() -> None:
    assert classify_refusal(answerable=True, refused=False) == "answered"
    assert classify_refusal(answerable=True, refused=True) == "false_refusal"
    assert classify_refusal(answerable=False, refused=True) == "refusal_correct"
    assert classify_refusal(answerable=False, refused=False) == "wrong_refusal"
