"""Length switching, semantic candidates, resume and statistical correctness; offline."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dialogue_generate as dg
import length_analysis as la


class Client:
    def __init__(self, job, fail=None, early=None):
        self.job, self.fail, self.early = job, fail, early
        self.requests = []

    def call(self, role, messages, validator=None):
        n = len(self.job['messages']) // 2 + 1
        self.requests.append((role, n, copy.deepcopy(messages)))
        if self.fail == (role, n):
            raise RuntimeError('interrupted')
        self.job['attempts'].extend([
            {'role': role, 'round': n, 'ok': False, 'usage': {'completion_tokens': 999}},
            {'role': role, 'round': n, 'ok': True, 'usage': {'completion_tokens': n*2}}])
        if role == 'character':
            return '答' * (n*3) + ' \n'
        end = self.early == n
        if validator is dg.validate_aligned_user:
            result = {'core_intent': '请求解释', 'minimal': '请解释。',
                      'long': '我想理解你这句话的意思。希望你解释一下这句话。', 'goal_completed': end}
        else:
            result = {'message': '问' * n, 'goal_completed': end}
        validator(result)
        return result


class ExperimentTests(unittest.TestCase):
    def job(self, condition='minimal_long', mode='natural'):
        catalog = dg.load_catalog(dg.DEFAULT_USERS, dg.DEFAULT_SCHEMES,
                                  dg.ROOT / 'profiles/Character_profile', conversation_mode='open_chat')
        job = dg.make_jobs(catalog, 1, 42, 'open_chat')[0]
        job['user_behavior'] = {'schema_version': '3.0', 'id': condition, 'tone': 'neutral',
                                'length_condition': condition, 'content_mode': mode}
        return job

    def test_schedules_and_twenty_rounds(self):
        for condition in dg.LENGTH_CONDITIONS:
            with self.subTest(condition=condition):
                job = self.job(condition)
                client = Client(job)
                dg.run_job(job, client, lambda: None)
                self.assertEqual(len(job['messages']), 40)
                self.assertEqual(job['stop_reason'], 'max_turns')
                first, second = condition.split('_')
                self.assertEqual([x['length_metrics']['planned_user_length'] for x in job['checks']],
                                 [first]*10 + [second]*10)
                self.assertIn('"current_response_length": "'+second+'"', client.requests[20][2][0]['content'])
                self.assertTrue(job['analysis_metrics']['full_20_rounds'])

    def test_early_stop_not_complete(self):
        job = self.job()
        dg.run_job(job, Client(job, early=5), lambda: None)
        stats = job['analysis_metrics']
        self.assertEqual(stats['completed_rounds'], 5)
        self.assertEqual(stats['second_half_rounds'], 0)
        self.assertFalse(stats['full_20_rounds'])
        self.assertIsNone(stats['character_half_change_chars']['delta_second_minus_first'])

    def test_aligned_resume_at_switch_and_isolation(self):
        job = self.job(mode='semantic_aligned')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save = lambda: dg.save_job(output, job)
            with self.assertRaises(RuntimeError):
                dg.run_job(job, Client(job, fail=('character', 11)), save)
            restored = dg.read_job(output, job['id'])
            self.assertEqual(restored['phase'], 'character')
            self.assertEqual(restored['messages'][-1]['content'], restored['semantic_drafts']['11']['long'])
            self.assertEqual(restored['messages'][18]['content'], restored['semantic_drafts']['10']['minimal'])
            client = Client(restored)
            dg.run_job(restored, client, lambda: dg.save_job(output, restored))
            self.assertEqual(client.requests[0][:2], ('character', 11))
            character = dg.actor_messages(restored, 'character')
            self.assertEqual(character[0]['content'], dg.CHARACTER_PROMPT+'\n'+dg.dumps({'profile': restored['character']['profile']}))
            for turn in character[1:]:
                self.assertNotIn('core_intent', turn['content'])
                self.assertNotIn('current_response_length', turn['content'])
            user = dg.actor_messages(restored, 'user')
            self.assertEqual(json.loads(user[1]['content']), restored['semantic_drafts']['1'])
            dg.export_jobs(output)
            training = json.loads((output/'training.jsonl').read_text(encoding='utf-8'))
            self.assertNotIn('core_intent', str(training))
            self.assertEqual(dg.read_job(output, restored['id'])['analysis_metrics'], restored['analysis_metrics'])

    def test_metrics_known_values_and_tokens_exclude_failed_retries(self):
        job = self.job()
        dg.run_job(job, Client(job), lambda: None)
        stats = job['analysis_metrics']
        self.assertEqual(stats['character_average_chars'], 31.5)
        self.assertEqual(stats['character_average_completion_tokens'], 21)
        self.assertEqual(stats['character_length_amplitude_chars'], 57)
        self.assertEqual(stats['character_half_change_chars']['delta_second_minus_first'], 30)
        self.assertEqual(stats['same_round']['n'], 20)
        self.assertEqual(stats['next_round']['n'], 19)
        self.assertEqual(stats['same_round']['pairs'][0], [1, 3])
        self.assertEqual(stats['next_round']['pairs'][0], [1, 6])
        self.assertAlmostEqual(stats['same_round']['pearson_r'], 1)
        self.assertAlmostEqual(stats['next_round']['pearson_r'], 1)
        self.assertIsNone(la.correlation([(1, 3), (1, 4)]))
        job.pop('turn_usage')
        self.assertIsNone(la.summarize_job(job)['character_average_completion_tokens'])

    def test_config_override_and_pair_materials(self):
        with tempfile.TemporaryDirectory() as directory:
            materials = []
            for condition in dg.LENGTH_CONDITIONS:
                output = Path(directory)/condition
                with patch('sys.argv', ['dialogue_generate.py', 'plan', '--count', '4',
                                        '--user-behavior', condition, '--output', str(output)]), \
                     patch('sys.stdout', io.StringIO()), patch.object(dg, 'load_api_env'):
                    dg.main()
                jobs = [dg.read_job(output, x) for x in dg.job_ids(output)]
                self.assertTrue(all(x['user_behavior']['length_condition'] == condition for x in jobs))
                materials.append([(x['user']['id'], x['character']['id'], x['topic'], x['seed_situation']) for x in jobs])
            self.assertTrue(all(x == materials[0] for x in materials))

    def test_balanced_quota_and_invalid_configs(self):
        catalog = dg.load_catalog(dg.DEFAULT_USERS, dg.DEFAULT_SCHEMES, dg.ROOT/'profiles/Character_profile')
        jobs, summary = dg.make_quota_jobs(catalog, 100, 42, dg.load_sampling_config(),
                                           dg.load_user_behaviors(dg.DEFAULT_USER_BEHAVIORS))
        self.assertEqual(summary['actual_counts']['length_condition'], dict.fromkeys(dg.LENGTH_CONDITIONS, 25))
        self.assertEqual(summary['target_counts'], summary['actual_counts'])
        with self.assertRaises(ValueError):
            dg.load_user_behaviors(dg.DEFAULT_USER_BEHAVIORS, 'sharp_minimal')
        for value in ({}, {'core_intent':'x', 'minimal':'x', 'long':'', 'goal_completed':False}):
            with self.assertRaises(ValueError):
                dg.validate_aligned_user(value)

    def test_group_baselines_exclude_incomplete(self):
        jobs = []
        for condition in dg.LENGTH_CONDITIONS:
            job = self.job(condition)
            dg.run_job(job, Client(job), lambda: None)
            jobs.append(job)
        early = self.job()
        dg.run_job(early, Client(early, early=2), lambda: None)
        report = la.report_jobs(jobs+[early])
        self.assertEqual(len(report['groups']), 4)
        self.assertEqual(len(report['baseline_contrasts']), 2)
        self.assertEqual(report['baseline_contrasts'][0]['difference_in_half_deltas_chars'], 0)

    def test_recursive_report_cli_does_not_call_models_or_change_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = {}
            for condition in ('minimal_minimal', 'minimal_long'):
                job = self.job(condition)
                dg.run_job(job, Client(job), lambda: None)
                dg.save_job(root/condition, job)
                transcript = root/condition/(job['id']+'.json')
                original[transcript] = transcript.read_bytes()
            with patch('sys.argv', ['length_analysis.py', '--output', str(root)]), \
                 patch('sys.stdout', io.StringIO()), \
                 patch.object(dg.ChatClient, 'call', side_effect=AssertionError('offline only')):
                la.main()
            report = dg.read_json(root/'length_analysis.json')
            self.assertEqual(len(report['dialogues']), 2)
            self.assertEqual(len(report['baseline_contrasts']), 1)
            self.assertEqual(report['dialogues'][0]['next_round']['n'], 19)
            self.assertEqual({p: p.read_bytes() for p in original}, original)


if __name__ == '__main__':
    unittest.main()
