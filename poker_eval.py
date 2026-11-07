"""Original pure-Python Hold'em evaluator; cards are integers rank*4+suit.

Only public card information is encoded. Returns the same ordering tuple as
card_utils.evaluate, using rank counts and suit masks instead of 21 subsets.
"""


def encode(card):
    return (card.value - 2) * 4 + 'CDHS'.index(card.suit)


def straight(mask):
    # Bit zero is a deuce, bit twelve an ace.
    for high in range(12, 3, -1):
        if (mask >> (high - 4)) & 31 == 31:
            return high + 2
    if mask & 4111 == 4111:  # A2345
        return 5
    return 0


def rank(cards):
    """Best five of 5-7 distinct encoded cards. Caller ensures validity."""
    counts = [0] * 13
    suits = [0] * 4
    suit_counts = [0] * 4
    mask = 0
    for card in cards:
        r, s = card >> 2, card & 3
        counts[r] += 1
        suits[s] |= 1 << r
        suit_counts[s] += 1
        mask |= 1 << r
    flush = 0
    for s in range(4):
        if suit_counts[s] >= 5:
            flush = suits[s]
            st = straight(flush)
            if st:
                return (8, st)
            break
    quads = [r + 2 for r in range(12, -1, -1) if counts[r] == 4]
    if quads:
        q = quads[0]
        return (7, q, max(r + 2 for r in range(13) if counts[r] and r + 2 != q))
    trips = [r + 2 for r in range(12, -1, -1) if counts[r] >= 3]
    pairs = [r + 2 for r in range(12, -1, -1) if counts[r] >= 2]
    if trips and len(pairs) >= 2:
        return (6, trips[0], next(p for p in pairs if p != trips[0]))
    if flush:
        return (5, *[r + 2 for r in range(12, -1, -1) if flush & (1 << r)][:5])
    st = straight(mask)
    if st:
        return (4, st)
    if trips:
        t = trips[0]
        return (3, t, *[r + 2 for r in range(12, -1, -1) if counts[r] and r + 2 != t][:2])
    if len(pairs) >= 2:
        p, q = pairs[:2]
        return (2, p, q, max(r + 2 for r in range(13) if counts[r] and r + 2 not in (p, q)))
    if pairs:
        p = pairs[0]
        return (1, p, *[r + 2 for r in range(12, -1, -1) if counts[r] and r + 2 != p][:3])
    return (0, *[r + 2 for r in range(12, -1, -1) if counts[r]][:5])


def preflop_score(a, b):
    """Ordering heuristic, not a calibrated equity or a betting rule."""
    hi, lo = sorted((a // 4 + 2, b // 4 + 2), reverse=True)
    if hi == lo:
        return .47 + .027 * hi
    gap = hi - lo
    return (.20 + .021 * hi + .012 * lo + .025 * (a % 4 == b % 4)
            + .012 * (gap == 1) + .005 * (gap == 2)
            - .003 * max(0, gap - 3) + .025 * (hi == 14 and lo <= 9))


def features(hole, board):
    """Made-hand quality and drawing potential for range likelihoods.

    Paired boards alone are not credited as a player's made pair. Quality
    is deliberately coarse; it is not substituted for showdown equity.
    """
    result = rank(hole + board)
    cat = result[0]
    values = [c // 4 + 2 for c in board]
    hv = [c // 4 + 2 for c in hole]
    paired = sorted(set(values), reverse=True)
    connects = hv[0] == hv[1] or any(v in values for v in hv)
    if cat >= 4:
        # With a full board, everyone may play the board.
        quality = .97
        if len(board) == 5 and result == rank(board):
            quality = .43
    elif cat == 3:
        quality = .91 if connects else .54
    elif cat == 2:
        quality = .85 if len(set(values)) == len(values) or hv[0] == hv[1] else .69
        if not connects:
            quality = .28
    elif cat == 1 and connects:
        pair = result[1]
        if pair >= max(values):
            quality = .73 + .035 * (pair > max(values))
        elif len(paired) > 1 and pair >= paired[1]:
            quality = .55
        else:
            quality = .40
    else:
        quality = .12 + .07 * (max(hv) == 14) + .04 * (min(hv) >= 10)
    draw = 0.0
    if len(board) < 5:
        all_cards = hole + board
        for s in range(4):
            count = sum(c % 4 == s for c in all_cards)
            ours = [c // 4 + 2 for c in hole if c % 4 == s]
            if count == 4 and ours:
                draw = max(draw, .28 + .03 * (max(ours) == 14))
        vals = {c // 4 + 2 for c in all_cards}
        if 14 in vals:
            vals.add(1)
        for lo in range(1, 11):
            window = set(range(lo, lo + 5))
            if len(vals & window) == 4 and any(v in window for v in hv):
                draw = max(draw, .16)
    return quality, draw
