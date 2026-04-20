from scripts.build_query_short_claim_report import assign_failure_label, classify_query_bucket


def test_classify_query_bucket_covers_required_buckets() -> None:
    assert classify_query_bucket("Edgar Wright is a person.") == "ambiguous_name_query"
    assert classify_query_bucket("Harold Macmillan was born on February 20, 1894.") == "temporal_query"
    assert classify_query_bucket('Hot Right Now is mistakenly attributed to DJ Fresh.') == "quoted_or_attribution_heavy"
    assert classify_query_bucket("Saxony is in Ireland.") == "very_short_factoid"


def test_assign_failure_label_prioritizes_no_cross_source_signal() -> None:
    label = assign_failure_label(
        cross_source_pair_count=0,
        short_query_claim=True,
        query_collapsed=True,
        evidence_noisy=True,
        query_bucket="very_short_factoid",
        query_tokens=4,
    )

    assert label == "NO_CROSS_SOURCE_SIGNAL"
