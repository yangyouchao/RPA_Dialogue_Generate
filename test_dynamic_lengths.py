"""Compatibility after replacing autonomous lengths with controlled schedules."""
import io
import json
import unittest
from unittest.mock import patch
import dialogue_generate as dg


class LengthContractTests(unittest.TestCase):
    def test_natural_response_does_not_require_model_length_label(self):
        dg.validate_user({'message': 'ok', 'goal_completed': False})
        with self.assertRaises(ValueError):
            dg.validate_user({'message': '', 'goal_completed': False})

    def test_old_behavior_uses_fixed_length_and_old_prompt(self):
        job = {'prompt_version': '2.0', 'user_behavior': {'response_length': 'long'}}
        self.assertEqual(dg.planned_user_length(job, 1), 'long')
        self.assertEqual(dg.planned_user_length(job, 11), 'long')
        self.assertEqual(dg.user_prompt(job), dg.LEGACY_USER_PROMPT)
        self.assertIn('user_behavior.response_length', dg.user_prompt(job))
        self.assertEqual(dg.stop_reasons({'messages': [{}]*16}), ['max_turns'])
        self.assertEqual(dg.stop_reasons({'messages': [{}]*16, 'max_rounds': 20}), [])

    def test_aligned_response_retry_keeps_billing_but_rejects_bad_candidate(self):
        config = dg.read_json(dg.ROOT / 'dialogue_config.json')
        config.update(min_interval_seconds=0, retries=1)
        contents = [{'message': 'bad format', 'goal_completed': False},
                    {'core_intent': 'explanation', 'minimal': 'why?', 'long': 'Please explain why.',
                     'goal_completed': False}]
        responses = [io.BytesIO(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(content)}}], 'usage': {'total_tokens': 10}}).encode()) for content in contents]
        audit = []
        with patch.object(dg.ChatClient, 'endpoint', return_value=({'json_mode': True}, 'https://example.invalid', 'mock', 'test')), patch('urllib.request.urlopen', side_effect=responses), patch('time.sleep'):
            result = dg.ChatClient(config, audit.append).call('user', [], dg.validate_aligned_user)
        self.assertEqual(result, contents[1])
        self.assertEqual([x['ok'] for x in audit], [False, True])
        self.assertEqual(sum(x['usage']['total_tokens'] for x in audit), 20)


if __name__ == '__main__':
    unittest.main()
