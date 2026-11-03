import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class BridgeTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('bridge', ROOT/'scripts/llm_bridge.py')
        self.bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bridge)

    def test_role_model_and_key_are_separate(self):
        cfg = {'base_url': 'https://maas.qianwenaiapi.com/compatible-mode/v1',
               'roles': {'reviewer': {'model': 'glm-5.3-prime', 'key_env': 'TEST_ARIS_GLM'}}}
        with patch.dict(os.environ, {'TEST_ARIS_GLM': 'test-only-secret'}):
            req = self.bridge.build_request(cfg, 'reviewer', 'hello', {})
        self.assertEqual(json.loads(req.data)['model'], 'glm-5.3-prime')
        self.assertEqual(req.full_url, cfg['base_url']+'/chat/completions')
        self.assertNotIn(b'test-only-secret', req.data)

    def test_missing_key_fails_closed(self):
        cfg = {'base_url': 'https://maas.qianwenaiapi.com/compatible-mode/v1',
               'roles': {'planner': {'model': 'deepseek-v4.1-flash-1017', 'key_env': 'NONEXISTENT_ARIS_TEST'}}}
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                self.bridge.build_request(cfg, 'planner', 'hello', {})

    def test_configured_reasoning_effort_forwarded(self):
        cfg = {'base_url': 'https://maas.qianwenaiapi.com/compatible-mode/v1',
               'roles': {'reviewer': {'model': 'glm-5.3-prime', 'key_env': 'TEST', 'reasoning_effort': 'low'}}}
        req=self.bridge.build_request(cfg,'reviewer','review',{'TEST':'dummy'})
        self.assertEqual(json.loads(req.data)['reasoning_effort'],'low')

    def test_unapproved_host_fails_before_credentials(self):
        cfg = {'base_url': 'https://unapproved.example/v1', 'roles': {}}
        with self.assertRaises(ValueError):
            self.bridge.build_request(cfg, 'planner', 'hello', {})

    def test_secret_file_permissions(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            p=Path(tmp)/'credentials.json'
            p.write_text('{"TEST":"dummy"}', encoding='utf-8')
            if os.name!='nt':
                p.chmod(0o644)
                with self.assertRaises(ValueError): self.bridge.read_secrets(p)
                p.chmod(0o600)
            self.assertEqual(self.bridge.read_secrets(p)['TEST'], 'dummy')

    def test_model_output_does_not_count_as_gpu_authorization(self):
        cfg=json.loads((ROOT/'.aris/project.json').read_text(encoding='utf-8'))
        self.assertFalse(cfg['execution']['gpu_launch_authorized'])
        self.assertEqual(cfg['execution']['max_concurrent_gpu_jobs'],1)
        self.assertEqual(cfg['evaluation']['topk'],5)
        self.assertEqual(cfg['evaluation']['filter_small'],50)


if __name__=='__main__': unittest.main()
