"""Offline no-limit Hold'em bot for the Manchester AI Society hackathon.

Original implementation. Only model.py and poker_eval.py are submission
sources. No I/O or external services are used during play. Range inference
is a conservative heuristic, not a solver or a claim of optimal play.
"""

from bisect import bisect_left
import math
import random
import time

from card_utils import Card
from game_state import GameStateView
from player_action import PlayerAction
from poker_eval import encode, features, preflop_score, rank


class Model:
    def __init__(self):
        self.hand = []
        self.player_id = None
        # Store the random state as plain data; no runtime resources persist.
        self.rng = random.Random(918273)
        combos = [(a, b, preflop_score(a, b)) for a in range(52) for b in range(a + 1, 52)]
        scores = sorted((s for _, _, s in combos), reverse=True)
        # Give tied hand classes the same percentile, regardless of suits.
        percentiles = {s: sum(v > s for v in scores) / len(scores) for s in set(scores)}
        self.combos = [(a, b, percentiles[s]) for a, b, s in combos]
        self.percentiles = {(a, b): p for a, b, p in self.combos}
        self.opponents = {}
        self.hands_played = 0
        self.fallbacks = 0
        self.max_seconds = 0.0
        self.last_equity = 0.0
        self.last_samples = 0
        self.last_se = 0.0

    def seed(self, hand: list[Card], player_id: int):
        self.hand = list(hand)
        self.hole = sorted(encode(c) for c in hand)
        self.player_id = player_id
        self.history = {}
        self.board_seen = []
        self.ledger_bets = []
        self.ledger_stacks = []
        self.ledger_pot = 0
        self.street_raises = 0
        self.preflop_raises = 0
        self.last_aggressor = None
        self.pf_flags = {}
        self.range_cache = {}
        self.finished = False

    def _stats(self, name):
        if name not in self.opponents:
            if len(self.opponents) >= 128:
                del self.opponents[next(iter(self.opponents))]
            self.opponents[name] = dict(hands=0, vpip=0, pfr=0, jams=0,
                                        responses=0, folds=0, calls=0,
                                        post_actions=0, post_bets=0)
        return self.opponents[name]

    def _profile(self, name):
        s = self._stats(name)
        n = s['hands']
        # Priors dominate small samples. All adjustments are bounded.
        vpip = (s['vpip'] + 12 * .58) / (n + 12)
        pfr = (s['pfr'] + 16 * .30) / (n + 16)
        folds = (s['folds'] + 12 * .35) / (s['responses'] + 12)
        aggro = (s['post_bets'] + 16 * .35) / (s['post_actions'] + 16)
        jam = (s['jams'] + 20 * .04) / (n + 20)
        return vpip, pfr, folds, aggro, jam

    def _consume(self, actions, g):
        """Replay just the delivered delta, never the already-updated view.

        A small public ledger supplies the price/pot before each event for
        response opportunities. Blind amounts are inferred from public
        bets plus delivered contributions on the first callback, including
        a blind posted all-in short. No hidden state is inspected.
        """
        if not self.ledger_bets:
            # The first decision is preflop: stacks + bets gives each
            # starting stack, even after delivered raises and short calls.
            initial = [s + b for s, b in zip(g.stacks, g.bets)]
            bets = [0] * g.num_players
            n = g.num_players
            sb = g.button if n == 2 else (g.button + 1) % n
            bb = (sb + 1) % n
            bets[sb] = min(g.small_blind, initial[sb])
            bets[bb] = min(g.big_blind, initial[bb])
            self.ledger_bets = bets
            self.ledger_stacks = [s - b for s, b in zip(initial, bets)]
            self.ledger_pot = sum(bets)
        for a in actions:
            p = a.player_id
            if a.kind == 'reveal_card':
                if p == -1:
                    self.board_seen.append(encode(a.card))
                    if len(self.board_seen) in (3, 4, 5):
                        self.ledger_bets = [0] * g.num_players
                        self.street_raises = 0
                continue
            if a.kind not in ('fold', 'call', 'raise') or p < 0:
                continue
            old = max(self.ledger_bets)
            cost = min(self.ledger_stacks[p], max(0, old - self.ledger_bets[p]))
            fraction = cost / max(1, self.ledger_pot)
            if a.kind == 'raise':
                fraction = (a.amount - old) / max(1, self.ledger_pot + cost)
            if p != self.player_id:
                name = g.teams[p]
                stats = self._stats(name)
                flags = self.pf_flags.setdefault(name, [False, False, False])
                if len(self.board_seen) < 3:
                    if a.kind == 'raise' or (a.kind == 'call' and cost > 0):
                        flags[0] = True
                    if a.kind == 'raise':
                        flags[1] = True
                        # A large opening bet, not a name-based opponent test.
                        flags[2] |= a.amount >= 12 * g.big_blind
                else:
                    stats['post_actions'] += 1
                    stats['post_bets'] += a.kind == 'raise'
                    if cost > 0:
                        stats['responses'] += 1
                        stats['folds'] += a.kind == 'fold'
                        stats['calls'] += a.kind == 'call'
                event = (tuple(self.board_seen), a.kind, fraction,
                         cost > 0, self.preflop_raises if not self.board_seen else self.street_raises)
                events = self.history.setdefault(p, [])
                # Bounded memory even in unusual very deep-stack hands.
                if len(events) < 32:
                    events.append(event)
            if a.kind == 'raise':
                added = a.amount - self.ledger_bets[p]
                self.ledger_pot += added
                self.ledger_stacks[p] -= added
                self.ledger_bets[p] = a.amount
                self.street_raises += 1
                self.last_aggressor = p
                if len(self.board_seen) < 3:
                    self.preflop_raises += 1
            elif a.kind == 'call':
                self.ledger_pot += cost
                self.ledger_stacks[p] -= cost
                self.ledger_bets[p] += cost
        # Trust the organiser's public snapshot over the approximate ledger.
        self.ledger_bets = list(g.bets)
        self.ledger_stacks = list(g.stacks)
        self.ledger_pot = g.pot

    def run(self, actions: list[PlayerAction], game_state: GameStateView) -> PlayerAction:
        start = time.perf_counter()
        g = game_state
        fallback = PlayerAction.call(self.player_id) if g.call_amount == 0 else PlayerAction.fold(self.player_id)
        try:
            self._consume(actions, g)
            # Budget the entire decision to well below the 10-second limit.
            action = self._decide(g, start + .65)
            g.validate_action(action)
        except (ArithmeticError, ValueError, IndexError, KeyError, TimeoutError):
            self.fallbacks += 1
            action = fallback
        self.max_seconds = max(self.max_seconds, time.perf_counter() - start)
        return action

    def _raise(self, g, total):
        limits = g.raise_range
        if not g.can_raise or limits is None or 'raise' not in g.legal_actions:
            return PlayerAction.call(self.player_id)
        low, high = limits
        return PlayerAction.raise_to(self.player_id, min(high, max(low, int(round(total)))))

    def _range(self, p, board, g, deadline):
        if time.perf_counter() >= deadline:
            raise TimeoutError('range budget exhausted')
        events = self.history.get(p, [])
        profile = self._profile(g.teams[p])
        key = (p, tuple(board), tuple(events), profile, g.num_players)
        if key in self.range_cache:
            return self.range_cache[key]
        known = set(self.hole + board)
        vpip, pfr, folds, aggro, jam = profile
        stats = self._stats(g.teams[p])
        learned = stats['hands'] >= 12
        wide = learned and (pfr > .55 or jam > .28)
        candidates, cumulative = [], []
        total = 0.0
        # Likelihoods have a nonzero floor to admit bluffs and model error.
        for index, (a, b, percentile) in enumerate(self.combos):
            if index % 32 == 0 and time.perf_counter() >= deadline:
                # An incomplete range biases rank/suit ordering. Abandon
                # the decision so run() returns the cheap legal fallback.
                raise TimeoutError('range budget exhausted')
            if a in known or b in known:
                continue
            weight = 1.0
            hole = [a, b]
            for past_board, kind, fraction, facing, raises in events:
                if kind == 'fold':
                    continue
                if len(past_board) < 3:
                    if kind == 'raise':
                        if raises == 0:
                            width = max(.16, min(.80, pfr * 1.5))
                        elif raises == 1:
                            width = max(.07, min(.45, pfr * .48))
                        else:
                            width = max(.035, min(.28, pfr * .23))
                        if g.num_players > 2:
                            width *= .7
                        if fraction > 2 and not wide:
                            width *= .70
                        if wide:
                            width = max(width, min(.97, pfr * 1.1 + jam * .3))
                        likelihood = .07 + .93 / (1 + math.exp(min(30, (percentile - width) / .055)))
                    elif facing:
                        width = min(.95, max(.30, vpip + .12))
                        if raises >= 2 and not wide:
                            width *= .55
                        likelihood = .16 + .84 / (1 + math.exp(min(30, (percentile - width) / .09)))
                    else:
                        likelihood = 1.0
                    weight *= likelihood
                else:
                    quality, draw = features(hole, list(past_board))
                    power = min(1, quality + draw * .8)
                    if kind == 'raise':
                        cutoff = .52 + .10 * (fraction > .8) + .10 * (raises > 0)
                        bluff = .09 + .22 * aggro
                        likelihood = bluff + (1 - bluff) / (1 + math.exp((cutoff - power) * 11))
                        # Repeated aggression is evidence, tempered for maniacs.
                        if len(past_board) == 5 and fraction > .7:
                            likelihood *= .4 + .6 * power
                    elif facing:
                        cutoff = .25 + min(.25, fraction * .35)
                        likelihood = .12 + .88 / (1 + math.exp((cutoff - power) * 9))
                    else:
                        likelihood = 1.0 - .35 * max(0, power - .55)
                    # Hyperaggressive players supply weaker range evidence.
                    weight *= likelihood ** (.55 if aggro > .65 else 1.0)
            total += max(weight, .0001)
            candidates.append((a, b))
            cumulative.append(total)
        answer = candidates, cumulative
        if len(self.range_cache) >= 16:
            self.range_cache.clear()
        self.range_cache[key] = answer
        return answer

    def _equity(self, g, deadline):
        board = [encode(c) for c in g.board]
        opponents = [p for p in g.players_in_hand if p != self.player_id]
        ranges = [self._range(p, board, g, deadline) for p in opponents]
        unseen = [c for c in range(52) if c not in self.hole and c not in board]
        missing = 5 - len(board)
        target = 480 if len(opponents) == 1 else 240
        if missing == 0:
            target = 800
        won = squared = 0.0
        n = 0
        for i in range(target):
            if i % 32 == 0 and time.perf_counter() >= deadline:
                if n:
                    break
                raise TimeoutError('equity budget exhausted')
            used = set()
            holes = []
            for candidates, cdf in ranges:
                # Condition each opponent range on previous sampled holes.
                for attempt in range(32):
                    a, b = candidates[bisect_left(cdf, self.rng.random() * cdf[-1])]
                    if a not in used and b not in used:
                        break
                else:
                    a, b = self.rng.sample([c for c in unseen if c not in used], 2)
                used.update((a, b))
                holes.append([a, b])
            runout = self.rng.sample([c for c in unseen if c not in used], missing) if missing else []
            full_board = board + runout
            mine = rank(self.hole + full_board)
            theirs = [rank(h + full_board) for h in holes]
            best = max(theirs, default=mine)
            share = 1.0 if mine > best else 1 / (1 + theirs.count(mine)) if mine == best else 0.0
            won += share
            squared += share * share
            n += 1
            if i % 32 == 31 and time.perf_counter() >= deadline:
                break
        equity = won / n
        se = math.sqrt(max(0, squared / n - equity * equity) / n)
        self.last_equity, self.last_samples, self.last_se = equity, n, se
        return equity, se

    def _decide(self, g, deadline):
        me = self.player_id
        count = len(g.players_in_hand)
        position = me == g.button
        bb = g.big_blind
        stack = min(g.stacks[me] + g.bets[me],
                    max((g.stacks[p] + g.bets[p] for p in g.players_in_hand if p != me), default=0))
        depth = stack / bb
        call = g.call_amount
        pot = g.pot
        opponents = [p for p in g.players_in_hand if p != me]
        profiles = [self._profile(g.teams[p]) for p in opponents]
        avg_fold = sum(p[2] for p in profiles) / max(1, len(profiles))
        aggro = sum(p[3] for p in profiles) / max(1, len(profiles))
        percentile = self.percentiles[tuple(self.hole)]
        hi, lo = sorted((c.value for c in self.hand), reverse=True)
        pair = hi == lo
        if g.street == 'preflop':
            if self.preflop_raises == 0:
                if count == 2:
                    if position:
                        # Open most buttons, mix limps with the lowest hands.
                        threshold = .84 if depth > 10 else .68
                        if percentile <= threshold:
                            if depth <= 7 and percentile < .46:
                                return self._raise(g, g.max_raise_to)
                            return self._raise(g, 2.25 * bb)
                        if call == 0 or (depth > 12 and self.rng.random() < .28):
                            return PlayerAction.call(me)
                        return PlayerAction.fold(me)
                    # The big blind can check its weak hands for free.
                    if percentile < .32:
                        return self._raise(g, (3.5 if depth >= 15 else 2.75) * bb)
                    return PlayerAction.call(me)
                # Multiway: tighten by players behind, retain sensible pairs.
                threshold = max(.12, .40 - .035 * (count - 3))
                if position:
                    threshold += .12
                if percentile < threshold:
                    return self._raise(g, 3 * bb + max(0, pot - 1.5 * bb))
                if call == 0 or (pair and depth > 18 and call <= bb):
                    return PlayerAction.call(me)
                return PlayerAction.fold(me)
            equity, se = self._equity(g, deadline)
            # Calling all-in realises all equity; deep calls lose some out
            # of position and when dominated. Avoid marginal noisy calls.
            all_in = call >= g.stacks[me] or not any(g.stacks[p] > 0 for p in opponents)
            margin = (.012 if all_in else .035 + .025 * (not position)) + min(.025, se * .6)
            raise_ratio = g.current_bet / bb
            premium = percentile < (.045 if self.preflop_raises >= 2 else .12)
            if g.can_raise and premium:
                if depth <= 30 or raise_ratio > depth * .22:
                    if equity > .54 + .03 * (self.preflop_raises >= 2):
                        return self._raise(g, g.max_raise_to)
                elif equity > .58:
                    return self._raise(g, g.current_bet * (3 if position else 3.5))
            if equity > g.pot_odds + margin:
                # Avoid excessively speculative calls into several players.
                if count > 2 and percentile > .48 and call > 3 * bb:
                    return PlayerAction.fold(me)
                return PlayerAction.call(me)
            return PlayerAction.call(me) if call == 0 else PlayerAction.fold(me)

        equity, se = self._equity(g, deadline)
        board = [encode(c) for c in g.board]
        quality, draw = features(self.hole, board)
        spr = g.spr(me)
        river = g.street == 'river'
        uncertainty = min(.035, .7 * se)
        all_in = call >= g.stacks[me] or not any(g.stacks[p] > 0 for p in opponents)
        if call:
            # Equity vs a betting range, with an extra realisation discount
            # on earlier streets; draws may continue at the right price.
            margin = .012 + uncertainty if river or all_in else .04 + uncertainty + .02 * (not position)
            if count > 2 and not all_in:
                margin += .025
            if g.can_raise:
                value_threshold = .77 if count == 2 else .82
                if equity > value_threshold + uncertainty and quality >= .69:
                    size = g.current_bet + .70 * (pot + call)
                    if spr < 1.5 or g.stacks[me] - (size - g.bets[me]) < .3 * (pot + call):
                        size = g.max_raise_to
                    return self._raise(g, size)
                # Rare semi-bluffs, with real drawing equity and chips behind.
                if (not river and count == 2 and draw >= .28 and equity > .39
                        and spr > 2.5 and self.street_raises <= 1
                        and self.rng.random() < .12 * min(1.5, avg_fold / .35)):
                    return self._raise(g, g.current_bet + .65 * (pot + call))
            if equity >= g.pot_odds + margin:
                return PlayerAction.call(me)
            return PlayerAction.fold(me)

        if not g.can_raise:
            return PlayerAction.call(me)
        # Value bet more often against callers, keep enough weak hands in
        # checks, and avoid treating a board-only made hand as a monster.
        value_threshold = .60 if river else .63
        if count > 2:
            value_threshold += .10
        if equity > value_threshold + uncertainty:
            fraction = .72 if equity > .80 else .50
            if draw > .15 or len({c % 4 for c in board}) <= 2:
                fraction = max(fraction, .65)
            if avg_fold < .22 and equity > .75:
                fraction = .9
            size = g.bets[me] + fraction * pot
            if spr < 1.25 and equity > .71:
                size = g.max_raise_to
            return self._raise(g, size)
        # Small continuation bets and delayed steals on suitable boards.
        # Bluff less multiway and into opponents who rarely fold.
        paired = len({c // 4 for c in board}) < len(board)
        dry = len({c % 4 for c in board}) >= 3 and not draw
        initiative = self.last_aggressor == me
        bluff_rate = .0
        if count == 2 and (position or initiative):
            bluff_rate = .12 if dry or paired else .05
            if draw >= .16 and not river:
                bluff_rate += .18
            bluff_rate *= min(1.6, max(.15, avg_fold / .35))
            if quality > .40 and draw == 0:
                bluff_rate *= .35  # prefer checking marginal showdown value
            if river and equity > .25:
                bluff_rate *= .25
        if self.rng.random() < bluff_rate:
            return self._raise(g, g.bets[me] + (.38 if not river else .60) * pot)
        return PlayerAction.call(me)

    def end_game(self, actions: list[PlayerAction], game_state: GameStateView):
        if self.finished:
            return
        self._consume(actions, game_state)
        for p, name in enumerate(game_state.teams):
            if p == self.player_id:
                continue
            stats = self._stats(name)
            flags = self.pf_flags.get(name, [False, False, False])
            stats['hands'] += 1
            stats['vpip'] += flags[0]
            stats['pfr'] += flags[1]
            stats['jams'] += flags[2]
        self.hands_played += 1
        self.finished = True
        self.range_cache.clear()
