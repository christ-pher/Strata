import json
import unittest
from serve.reasoning_guard import ReasoningRepetitionGuard
from serve import test_server as fixtures


class Detection(unittest.TestCase):
    def detects(self, tokens):
        guard = ReasoningRepetitionGuard()
        return any(guard.push(t) for t in tokens)

    def test_persistent_loop(self):
        self.assertTrue(self.detects(list(range(100)) * 30))

    def test_unique_long_reasoning(self):
        self.assertFalse(self.detects(range(20000)))

    def test_short_refrain_and_one_repeated_passage(self):
        self.assertFalse(self.detects(list(range(200)) * 2 + list(range(200, 3000))))
        tokens = []
        for i in range(100):
            tokens.extend(range(10))
            tokens.extend(range(100 + i * 50, 150 + i * 50))
        self.assertFalse(self.detects(tokens))

    def test_memory_is_bounded(self):
        guard = ReasoningRepetitionGuard()
        for t in range(10000):
            guard.push(t)
        self.assertEqual(len(guard.tokens), 2048)


class Intervention(unittest.TestCase):
    tearDown = fixtures.ThinkingBudget.tearDown
    post = fixtures.ThinkingBudget.post
    openai = fixtures.ThinkingBudget.openai

    def setUp(self):
        fixtures.ThinkingBudget.setUp(self)
        self.svc.reasoning_repetition_guard = True
        self.engine.THOUGHT = 'Let me reconsider exactly the same unresolved question again. ' * 40

    # Inherit the HTTP fixture, not the fixed-budget test cases.
    def test_loop_continues_to_answer(self):
        code, body = self.openai(max_tokens=3500)
        self.assertEqual(code, 200)
        choice = body['choices'][0]
        self.assertEqual(choice['message']['content'], fixtures.ThinkingEngine.ANSWER)
        self.assertEqual(choice['finish_reason'], 'stop')
        self.assertEqual(len(self.engine.prompts), 2)
        self.assertLess(len(choice['message']['reasoning_content']), len(self.engine.THOUGHT))

    def test_disabled_guard_preserves_reasoning(self):
        self.svc.reasoning_repetition_guard = False
        code, body = self.openai(max_tokens=3500)
        self.assertEqual(code, 200)
        self.assertEqual(body['choices'][0]['message']['reasoning_content'], self.engine.THOUGHT)
        self.assertEqual(len(self.engine.prompts), 1)

    def test_progressing_reasoning_is_not_capped(self):
        self.engine.THOUGHT = ''.join(f'Step {i}: examine case {i*i} and record value {i*17}.\n' for i in range(55))
        code, body = self.openai(max_tokens=3500)
        self.assertEqual(code, 200)
        self.assertEqual(body['choices'][0]['message']['reasoning_content'], self.engine.THOUGHT)
        self.assertEqual(len(self.engine.prompts), 1)

    def test_repeated_answer_is_not_interrupted(self):
        self.engine.THOUGHT = ''
        self.engine.ANSWER = 'Repeated answer text. ' * 100
        code, body = self.openai(max_tokens=3500)
        self.assertEqual(code, 200)
        self.assertEqual(body['choices'][0]['message']['content'], self.engine.ANSWER)
        self.assertEqual(len(self.engine.prompts), 1)

    def test_anthropic_stream_intervention(self):
        code, raw = self.post('/v1/messages', {
            'model': 'm', 'max_tokens': 3500, 'stream': True,
            'messages': [{'role': 'user', 'content': '2+2?'}]})
        self.assertEqual(code, 200)
        events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
        answer = ''.join(event.get('delta', {}).get('text', '') for event in events)
        self.assertEqual(answer, fixtures.ThinkingEngine.ANSWER)
        self.assertEqual(events[-2]['delta']['stop_reason'], 'end_turn')
        self.assertEqual(len(self.engine.prompts), 2)

    def test_insufficient_output_room_preserves_length_finish(self):
        code, body = self.openai(max_tokens=1024)
        self.assertEqual(code, 200)
        self.assertEqual(body['choices'][0]['finish_reason'], 'length')
        self.assertEqual(len(self.engine.prompts), 1)

    def test_request_cannot_override_enabled_config(self):
        code, body = self.openai(max_tokens=3500, reasoning_repetition_guard=False)
        self.assertEqual(code, 200)
        self.assertEqual(body['choices'][0]['message']['content'], fixtures.ThinkingEngine.ANSWER)
        self.assertEqual(len(self.engine.prompts), 2)

    def test_request_cannot_override_disabled_config(self):
        self.svc.reasoning_repetition_guard = False
        code, body = self.openai(max_tokens=3500, reasoning_repetition_guard=True)
        self.assertEqual(code, 200)
        self.assertEqual(body['choices'][0]['message']['reasoning_content'], self.engine.THOUGHT)
        self.assertEqual(len(self.engine.prompts), 1)

    def test_shared_settings_do_not_accept_config_only_flag(self):
        with self.assertRaises(ValueError):
            self.svc.set_shared({'reasoning_repetition_guard': False})
