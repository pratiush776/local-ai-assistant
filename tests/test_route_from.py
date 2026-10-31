from assistant.decide import route_from


def test_clear_winner_routes_local():
    scores = {"action": 0.80, "fact": 0.10, "reason": 0.05, "chat": 0.05}
    route, candidate = route_from(scores)

    assert route == "local"
    assert candidate == ["action"]


def test_two_way_tie_routes_ask_sorted_by_score():
    scores = {"action": 0.50, "fact": 0.30, "reason": 0.10, "chat": 0.10}
    route, candidate = route_from(scores)

    assert route == "ask"
    assert candidate == ["action", "fact"]


def test_three_way_tie_routes_ask_sorted_by_score():
    scores = {"action": 0.30, "fact": 0.30, "reason": 0.30, "chat": 0.10}
    route, candidate = route_from(scores)

    assert route == "ask"
    assert candidate == ["action", "fact", "reason"]


def test_weak_leader_routes_ask_with_single_candidate():
    scores = {"action": 0.64, "fact": 0.12, "reason": 0.12, "chat": 0.12}
    route, candidate = route_from(scores)
    assert route == "ask"
    assert candidate == ["action"]


def test_nothing_alive_routes_ask_empty():
    scores = {"action": 0.10, "fact": 0.10, "reason": 0.10, "chat": 0.10}
    route, candidate = route_from(scores)

    assert route == "ask"
    assert candidate == []
