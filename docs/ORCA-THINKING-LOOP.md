# Orca thinking-loop investigation, 2026-10-01

The strongest recent candidate is strata-orca-iq4_xs.log:1396:
70,751 prompt tokens, 32,000 generated tokens in 404.323 seconds.
Suffix lookup accepted 21,750 / 21,816 repeated-context draft tokens.
This is consistent with repetitive generation reaching a 32,000-token
request cap, rather than a frozen engine. The engine log does not retain
request sampling, reasoning text, or the API finish reason, so the specific
request and termination reason cannot be established conclusively.
The next logged request contained 102,802 prompt tokens and generated 9,752
more tokens with high suffix acceptance, consistent with the repetitive
output being carried into subsequent context. Earlier requests also show
many consecutive 4,803-token outputs as history grew to 176,049 tokens.

The saved configuration supplied neither sampling defaults nor a reasoning
budget. Requests without a temperature therefore decode greedily. Browser
chat supplies sampled settings by default, so greedy decoding cannot be
assumed for this incident. Thinking effort is a template hint, not a hard
budget. No dedicated automatic repetition detector was found in the server
run path. Suffix drafts are verified against target-model outputs before
acceptance; their high acceptance is evidence consistent with repetition,
not proof that speculation caused it. Numerical, quantization, long-context,
and cache effects have not been excluded through an exact request replay.
Configured context is 262,144; 32,768 resident KV cells is a streaming window,
not the maximum context. The candidate request was below the context limit.
No runtime failure is logged alongside it. Full-context quality remains
unvalidated in the existing Orca IQ4_XS notes.

Mitigation applied to the local saved config:

- sampling defaults: temperature 0.6, top_p 0.95, top_k 20;
- reasoning_budget_tokens: 8192.

The existing server budget implementation stops generation at a clean token
boundary, appends a wrap-up plus </think>, and resumes to produce an answer
within the remaining total output budget. It contains excessive thinking;
it does not guarantee answer quality or prevent repetitive answer text.
The server was not running at inspection, so changes take effect on next
normal launch. Explicit client sampling fields override defaults; an explicit
reasoning_budget_tokens=0 disables the budget. No model startup/replay was
performed. Existing ThinkingBudget, SamplingKeys, and SharedSettings tests
passed (17 tests); config parsing also passed.

For recovery, discard the repetitive assistant response or start a fresh
chat using a concise summary of useful context. Retry with sampling enabled,
medium/low thinking, and sufficient total output tokens to leave room for
an answer after the thinking budget. If recurrence persists, preserve the
exact request/settings and compare speculation disabled (--spec 0 and
--suffix-draft 0), prompt reuse disabled, and the experimental speed projection
disabled, one setting at a time. These are diagnostic comparisons, not proven
fixes.

General Qwen thinking guidance (not an Orca-specific validation):
https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html
warns against greedy decoding due to endless repetition and recommends
0.6 / 0.95 / 20 for Qwen3 thinking.

## Repetition intervention replaces the cap, 2026-10-02

At the user's request, Orca's 8192-token thinking cap was removed and
`"reasoning_repetition_guard": true` was enabled. Sampled defaults remain.
The guard only tracks newly generated thinking tokens, retaining at most
2048 tokens and checking every 64 tokens after at least 1024 tokens. It
intervenes when at least 65% of 32-token spans in the window have already
appeared earlier in that window. It then uses the same clean-boundary
wrap-up and answer continuation as the existing budget implementation.
No fixed thinking length is imposed. Other configs remain opt-out.

This heuristic detects exact token repetition, including longer cycles;
it does not detect all semantic/paraphrased loops. Legitimate repetitive
thinking can trigger it. It never inspects ordinary answer tokens, does
not remove the original repeated text from the prompt, and does not override
client output limits or context capacity. Insufficient remaining output
room still ends with `length`. Explicit request thinking budgets still work.
Server startup logs confirm activation, and intervention logs say
`repeated thinking detected: wrapping up the thinking`.
The server was not running when the change was made; next launch loads it.
Tests use mock model generation, not a replay of the original Orca task.

Validation: all 92 tests in `serve.test_reasoning_guard` and
`serve.test_server` passed, including OpenAI answer continuation, Anthropic
streaming, nonrepetitive reasoning, repeated answer text, disabled guard,
insufficient output room, and existing thinking budgets. Script startup,
Python syntax checks and `git diff --check` passed. A local detector-only
benchmark over 32000 tokens measured 0.75s for varied tokens and 0.55s for
repeated tokens; this is not an end-to-end Orca performance measurement.

Control is config-only: `"reasoning_repetition_guard": true` enables it;
`false` or omission disables it. The UI/runtime toggle was removed at the
user's request. Browser and API request settings cannot override the model
config. Restart the server after editing the config.
