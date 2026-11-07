"""Extra bot checks. Run: .venv/bin/python -m pytest test_bot.py -q

Engine/public-contract tests are unchanged. These checks test the new bot,
its simulator, event consumption and evaluator against independent evidence.
"""
import pickle
import random
import time

import pytest

from card_utils import Card, evaluate, full_deck
from engine import play_hand
from game_state import GameState, GameStateView
from model import Model
from player_action import PlayerAction
from poker_eval import encode, rank


def cards(text):
    return [Card.parse(s) for s in text.split()]


def test_fast_evaluator_matches_reference():
    rng = random.Random(617)
    deck = full_deck()
    for size in (5, 6, 7):
        for _ in range(1200):
            h = rng.sample(deck, size)
            assert rank([encode(c) for c in h]) == evaluate(h)
    for text in (
        'AS 2S 3S 4S 5S KC KD', 'AS AH AD KS KH KD 2C',
        '2S 2H 2D 2C AS KS QS', 'AS KS QS JS 9S 8S 7S',
        'AS AH KS KH QS QH 2C', 'AC 2D 3H 4S 5D 6C 7D',
    ):
        h = cards(text)
        assert rank([encode(c) for c in h]) == evaluate(h)


@pytest.mark.parametrize('total, expected', [(-100, 40), (40, 40), (500, 500), (10000, 500)])
def test_raise_totals_clamped(total, expected):
    g = GameState([500, 500])
    m = Model()
    m.seed(cards('AS AH'), 0)
    action = m._raise(GameStateView(g), total)
    assert action.amount == expected
    g.validate_action(action)


def test_closed_raising_after_short_all_in():
    g = GameState([500, 130, 500])
    events = [PlayerAction.raise_to(0, 100), PlayerAction.raise_to(1, 130), PlayerAction.call(2)]
    for a in events:
        g.process_action(a)
    assert g.to_act == 0 and not g.can_raise and g.raise_range is None
    m = Model()
    m.seed(cards('AS AH'), 0)
    m._consume(events, GameStateView(g))
    action = m.run([], GameStateView(g))
    assert action.kind == 'call'
    g.validate_action(action)
    assert m.fallbacks == 0


def test_short_stack_calls_and_no_raise():
    g = GameState([30, 500])
    events = [PlayerAction.call(0), PlayerAction.raise_to(1, 500)]
    for a in events:
        g.process_action(a)
    assert g.call_amount == 10 and not g.can_raise
    m = Model()
    m.seed(cards('AS AH'), 0)
    action = m.run(events, GameStateView(g))
    assert action.kind == 'call'
    g.validate_action(action)
    assert m.fallbacks == 0


def test_free_check_and_fallback(monkeypatch):
    g = GameState([500, 500])
    event = PlayerAction.call(0)
    g.process_action(event)
    m = Model()
    m.seed(cards('2S 7H'), 1)
    def failure(*args):
        raise ValueError('deliberate calculation failure')
    monkeypatch.setattr(m, '_decide', failure)
    action = m.run([event], GameStateView(g))
    assert action.kind == 'call' and action.player_id == 1
    g.validate_action(action)
    assert m.fallbacks == 1


def test_facing_bet_fallback(monkeypatch):
    g = GameState([500, 500])
    m = Model()
    m.seed(cards('AS AH'), 0)
    def failure(*args):
        raise ValueError('deliberate calculation failure')
    monkeypatch.setattr(m, '_decide', failure)
    action = m.run([], GameStateView(g))
    assert action.kind == 'fold' and action.player_id == 0
    g.validate_action(action)


def test_exhausted_sampling_budget_uses_legal_fallback(monkeypatch):
    g = GameState([500, 500])
    m = Model()
    m.seed(cards('AS AH'), 0)
    def expired(view, deadline):
        return m._equity(view, time.perf_counter() - 1)
    monkeypatch.setattr(m, '_decide', expired)
    action = m.run([], GameStateView(g))
    g.validate_action(action)
    assert action.kind == 'fold' and m.fallbacks == 1


def test_every_simulated_hand_has_distinct_known_cards(monkeypatch):
    import model
    original = model.rank
    seen = []
    def checked(h):
        assert len(h) == 7 and len(set(h)) == 7
        seen.append(h)
        return original(h)
    monkeypatch.setattr(model, 'rank', checked)
    g = GameState([500] * 5)
    m = Model()
    m.seed(cards('AS KH'), 3)
    m._equity(GameStateView(g), time.perf_counter() + 1)
    assert len(seen) >= 100
    for i in range(0, len(seen), 5):
        block = seen[i:i+5]
        assert block[0][:2] == m.hole
        assert all(h[2:] == block[0][2:] for h in block)
        assert len({c for h in block for c in h[:2]}) == 10


@pytest.mark.parametrize('players', [2, 3, 6, 9])
def test_board_only_ties_split_for_every_player(players):
    g = GameState([500] * players)
    g.board = cards('AS KS QS JS TS')
    g.street = 'river'
    m = Model()
    m.seed(cards('2C 3D'), 0)
    eq, se = m._equity(GameStateView(g), time.perf_counter() + 2)
    assert eq == pytest.approx(1 / players)
    assert se < 1e-7


def test_event_deltas_streets_and_team_names():
    m = Model()
    m.seed(cards('AS AH'), 0)
    g = GameState([500, 500], teams=['us', 'rival'])
    m._consume([], GameStateView(g))
    events = [PlayerAction.raise_to(0, 60), PlayerAction.call(1)]
    for a in events:
        g.process_action(a)
    reveal = [PlayerAction.reveal_card(c) for c in cards('2C 4D 8H')]
    for a in reveal:
        g.process_action(a)
    check = PlayerAction.call(1)
    g.process_action(check)
    m._consume(events + reveal + [check], GameStateView(g))
    assert m.preflop_raises == 1 and m.street_raises == 0
    assert len(m.board_seen) == 3
    assert len(m.history[1]) == 2
    assert m.ledger_pot == g.pot and m.ledger_bets == g.bets
    assert m.ledger_stacks == g.stacks
    # Fold our hand to end it; end_game consumes only the remaining delta.
    fold = PlayerAction.fold(0)
    g.process_action(fold)
    m.end_game([fold], GameStateView(g))
    assert m.opponents['rival']['hands'] == 1
    assert m.opponents['rival']['vpip'] == 1
    assert m.opponents['rival']['post_actions'] == 1
    m.seed(cards('KS KH'), 1)
    assert m.history == {} and m.preflop_raises == 0
    swapped = GameState([500, 500], teams=['rival', 'us'])
    m.end_game([], GameStateView(swapped))
    assert m.opponents['rival']['hands'] == 2
    assert 'us' not in m.opponents


def test_range_responds_to_strong_betting_without_becoming_certain():
    g = GameState([500, 500], teams=['us', 'rival'])
    m = Model()
    m.seed(cards('7C 2D'), 0)
    initial, _ = m._range(1, [], GameStateView(g), time.perf_counter() + 1)
    m.history[1] = [((), 'raise', 10.0, True, 1)]
    tight, cdf = m._range(1, [], GameStateView(g), time.perf_counter() + 1)
    weights = [cdf[0]] + [b - a for a, b in zip(cdf, cdf[1:])]
    weighted = sum(m.percentiles[h] * w for h, w in zip(tight, weights)) / cdf[-1]
    uniform = sum(m.percentiles[h] for h in initial) / len(initial)
    assert weighted < uniform - .1
    assert min(weights) > 0  # still admits weak hands/bluffs


def test_pickle_after_learning_and_real_play():
    from arena.dummies import RandomBot
    m = Model()
    other = RandomBot(seed=9)
    rng = random.Random(72)
    for i in range(20):
        deck = full_deck()
        rng.shuffle(deck)
        seat = i % 2
        models = [m, other] if seat == 0 else [other, m]
        names = ['us', 'rival'] if seat == 0 else ['rival', 'us']
        result = play_hand(models, deck, stacks=[500, 500], teams=names)
        assert not result.faults
        m = pickle.loads(pickle.dumps(m, protocol=4))
    assert m.opponents['rival']['hands'] == 20
    assert m.hands_played == 20 and m.fallbacks == 0
