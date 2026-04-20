# P3 Query Short-Claim Report

## Scope
- Baseline configuration: chunking v2 enabled, evidence hygiene implemented but disabled by default.
- Fixed regression harness: 30 FEVER-dev records.
- Larger diagnostic slice: 150 FEVER-dev records.
- No handoff schema changes and no P1/P2 core logic changes.

## Decisive NLI Rate By Query Bucket

### Fixed 30-record regression slice
| Bucket | Records | Short-query rate | Decisive NLI rate | Avg cross-source pairs | Avg evidence penalty |
| --- | ---: | ---: | ---: | ---: | ---: |
| very_short_factoid | 2 | 1.0000 | 0.0000 | 0.5000 | 0.1000 |
| yes_no_claim_like | 11 | 0.0000 | 0.0000 | 1.2727 | 0.1341 |
| entity_seeking | 8 | 0.1250 | 0.1250 | 2.1250 | 0.1125 |
| temporal_query | 6 | 0.3333 | 0.0000 | 3.1667 | 0.1000 |
| ambiguous_name_query | 2 | 1.0000 | 0.0000 | 1.0000 | 0.1500 |
| quoted_or_attribution_heavy | 1 | 0.0000 | 0.0000 | 2.0000 | 0.1000 |

### Larger 150-record diagnostic slice
| Bucket | Records | Short-query rate | Decisive NLI rate | Avg cross-source pairs | Avg evidence penalty |
| --- | ---: | ---: | ---: | ---: | ---: |
| very_short_factoid | 14 | 1.0000 | 0.0000 | 0.5000 | 0.1268 |
| yes_no_claim_like | 59 | 0.0000 | 0.0000 | 0.5763 | 0.1102 |
| entity_seeking | 41 | 0.2195 | 0.0244 | 0.7561 | 0.0951 |
| temporal_query | 32 | 0.0938 | 0.0000 | 0.8438 | 0.1211 |
| ambiguous_name_query | 2 | 1.0000 | 0.0000 | 1.0000 | 0.1500 |
| quoted_or_attribution_heavy | 2 | 0.0000 | 0.0000 | 1.0000 | 0.1000 |

## Top Recurring Failure Labels

### Fixed 30-record regression slice
- `Q_COLLAPSED_IN_P1`: 6
- `EVIDENCE_OK_BUT_NOT_COMPARABLE`: 6
- `Q_UNDERSPECIFIED`: 5
- `NO_CROSS_SOURCE_SIGNAL`: 5
- `Q_SHORT`: 4
- `EVIDENCE_STILL_NOISY`: 3

### Larger 150-record diagnostic slice
- `NO_CROSS_SOURCE_SIGNAL`: 93
- `EVIDENCE_OK_BUT_NOT_COMPARABLE`: 17
- `Q_COLLAPSED_IN_P1`: 11
- `Q_UNDERSPECIFIED`: 11
- `Q_SHORT`: 10
- `EVIDENCE_STILL_NOISY`: 7

## Failure Families
- Regression 30 query-side family: 15
- Regression 30 retrieval-side family: 8
- Regression 30 comparison-gap family: 6
- Diagnostic 150 query-side family: 32
- Diagnostic 150 retrieval-side family: 100
- Diagnostic 150 comparison-gap family: 17

## Concrete Bad Examples
- `137334` [temporal_query] `Fox 2000 Pictures released the film Soul Food.` -> `Q_COLLAPSED_IN_P1`; short_query=False; collapsed=True; cross_source_pairs=5; evidence_penalty=0.0000
- `111897` [entity_seeking] `Telemundo is a English-language television network.` -> `EVIDENCE_STILL_NOISY`; short_query=False; collapsed=False; cross_source_pairs=2; evidence_penalty=0.2000
- `89891` [temporal_query] `Damon Albarn's debut album was released in 2011.` -> `EVIDENCE_OK_BUT_NOT_COMPARABLE`; short_query=False; collapsed=False; cross_source_pairs=5; evidence_penalty=0.1000
- `181634` [entity_seeking] `There is a capital called Mogadishu.` -> `Q_UNDERSPECIFIED`; short_query=False; collapsed=False; cross_source_pairs=4; evidence_penalty=0.1000
- `219028` [yes_no_claim_like] `Savages was exclusively a German film.` -> `Q_UNDERSPECIFIED`; short_query=False; collapsed=False; cross_source_pairs=1; evidence_penalty=0.0750
- `108281` [yes_no_claim_like] `Andrew Kevin Walker is only Chinese.` -> `Q_UNDERSPECIFIED`; short_query=False; collapsed=False; cross_source_pairs=1; evidence_penalty=0.1000
- `204361` [temporal_query] `The Cretaceous ended.` -> `Q_COLLAPSED_IN_P1`; short_query=True; collapsed=True; cross_source_pairs=1; evidence_penalty=0.1000
- `54168` [entity_seeking] `Murda Beatz's real name is Marshall Mathers.` -> `NO_CROSS_SOURCE_SIGNAL`; short_query=False; collapsed=False; cross_source_pairs=0; evidence_penalty=0.1000
- `105095` [entity_seeking] `Nicholas Brody is a character on Homeland.` -> `Q_UNDERSPECIFIED`; short_query=False; collapsed=False; cross_source_pairs=2; evidence_penalty=0.0000
- `18708` [yes_no_claim_like] `Charles Manson has been proven innocent of all crimes.` -> `EVIDENCE_OK_BUT_NOT_COMPARABLE`; short_query=False; collapsed=False; cross_source_pairs=1; evidence_penalty=0.1000
- `90809` [yes_no_claim_like] `Sean Penn is only ever a stage actor.` -> `EVIDENCE_STILL_NOISY`; short_query=False; collapsed=False; cross_source_pairs=1; evidence_penalty=0.2000
- `204443` [temporal_query] `Brad Wilk helped co-found Rage in 1962.` -> `EVIDENCE_OK_BUT_NOT_COMPARABLE`; short_query=False; collapsed=False; cross_source_pairs=2; evidence_penalty=0.1000
- `192714` [entity_seeking] `CBS is the network that aired The Millers.` -> `Q_UNDERSPECIFIED`; short_query=False; collapsed=False; cross_source_pairs=2; evidence_penalty=0.1000
- `159706` [ambiguous_name_query] `Edgar Wright is a person.` -> `Q_SHORT`; short_query=True; collapsed=False; cross_source_pairs=1; evidence_penalty=0.2000
- `107786` [yes_no_claim_like] `Ann Richards was professionally involved in politics.` -> `NO_CROSS_SOURCE_SIGNAL`; short_query=False; collapsed=False; cross_source_pairs=0; evidence_penalty=0.2000

## Final Judgment
- The fixed 30-record harness points to a split picture: query-side collapse/shortness is substantial, but evidence-side noise still appears in a noticeable minority.
- On the 150-record slice, raw `NO_CROSS_SOURCE_SIGNAL` dominates because the diagnostic slice is much larger than the tiny matched wiki sample; that is best treated as a corpus-coverage artifact, not a clean retrieval-algorithm signal.
- After separating those zero-signal cases out, query-side failures (`Q_SHORT`, `Q_COLLAPSED_IN_P1`, `Q_UNDERSPECIFIED`) outnumber retrieval-side noise labels in the diagnosable subset.
- The next owner should therefore be closer to P1/query-understanding or query claim transformation than further P3 retrieval tuning.
