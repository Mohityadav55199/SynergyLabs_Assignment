from evaluation.judge import parse_judge_json, score_verdict

GOOD = '{"claims":[{"claim":"a","supported":true},{"claim":"b","supported":false}],"relevance":5,"correctness":3,"rationale":"x"}'

def test_plain_json():
    s = score_verdict(parse_judge_json(GOOD))
    assert s["faithfulness"] == 0.5 and s["relevance"] == 1.0 and s["correctness"] == 0.5

def test_code_fence_and_prose_wrapper():
    assert parse_judge_json("Here you go:\n```json\n" + GOOD + "\n```\nDone.") is not None

def test_reasoning_block_stripped():
    assert parse_judge_json("<think>{not json}</think>" + GOOD) is not None

def test_malformed_returns_none():
    assert parse_judge_json('{"claims": [') is None
    assert parse_judge_json("") is None
    assert parse_judge_json("no json at all") is None

def test_invalid_schema_rejected():
    assert score_verdict({"claims": [], "relevance": 9, "correctness": 3}) is None
    assert score_verdict({"relevance": 3}) is None

def test_no_claims_means_faithfulness_not_applicable():
    s = score_verdict({"claims": [], "relevance": 1, "correctness": 1})
    assert s["faithfulness"] is None
