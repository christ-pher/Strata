"""Bounded exact-token repetition detection for generated thinking only."""
from collections import deque


class ReasoningRepetitionGuard:
    """Intervene when >=65% of a 2K-token window repeats earlier 32-token spans.

    Wait for 1K tokens and check every 64 tokens. This deliberately tolerates
    short refrains and occasional quotes; it cannot detect paraphrased loops.
    Memory and check cost are independent of the total reasoning length.
    """

    def __init__(self):
        self.tokens = deque(maxlen=2048)
        self.count = 0

    def push(self, token: int) -> bool:
        self.tokens.append(token)
        self.count += 1
        if len(self.tokens) < 1024 or self.count % 64:
            return False
        tokens = list(self.tokens)
        seen = set()
        repeated = 0
        spans = len(tokens) - 32 + 1
        for i in range(spans):
            span = tuple(tokens[i:i + 32])
            if span in seen:
                repeated += 1
            else:
                seen.add(span)
        return repeated / spans >= 0.65
