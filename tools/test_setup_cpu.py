"""Native CPU build identity and targeted object invalidation regression checks."""
import tempfile
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch
import setup


class NativeCpuBuild(unittest.TestCase):
    def test_guest_feature_change_invalidates_identity(self):
        with patch.object(setup, 'WIN', True):
            with patch.object(setup, 'cpu_info', return_value=('EPYC', False, False)):
                old = setup.native_cpu_signature()
            with patch.object(setup, 'cpu_info', return_value=('EPYC', True, False)):
                self.assertNotEqual(old, setup.native_cpu_signature())
                self.assertEqual(setup.native_cpu_signature(), setup.native_cpu_signature())

    def test_linux_instruction_change_invalidates_identity(self):
        with patch.object(setup, 'WIN', False), patch.object(setup, 'cpu_info', return_value=('EPYC', True, False)):
            with patch.object(Path, 'read_text', return_value='flags : avx avx2 fma\n'):
                old = setup.native_cpu_signature()
            with patch.object(Path, 'read_text', return_value='flags : avx avx2 fma f16c\n'):
                self.assertNotEqual(old, setup.native_cpu_signature())

    def test_cpu_change_rebuilds_with_compiler_cache_bypassed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'engine').mkdir()
            (root / 'build').mkdir()
            (root / 'engine/strata').write_text('old')
            (root / 'build/strata').write_text('rebuilt')
            (root / 'engine/BUILD.json').write_text(json.dumps({
                'source': 'local', 'src': 'same-source', 'archs': [70],
                'cpu_signature': 'old-cpu'}))
            def compile_check(*args):
                self.assertEqual(os.environ['CCACHE_DISABLE'], '1')
            with patch.object(setup, 'ROOT', root), patch.object(setup, 'EXE', 'strata'), \
                 patch.object(setup, 'source_hash', return_value='same-source'), \
                 patch.object(setup, 'native_cpu_signature', return_value='new-cpu'), \
                 patch.object(setup, 'install_build_tools', return_value=('/usr/local/cuda/bin/nvcc', None)), \
                 patch.object(setup, 'find_nvcc', return_value=('/usr/local/cuda/bin/nvcc', 12)), \
                 patch.object(setup, 'cuda_host_compiler', return_value=[]), \
                 patch.object(setup, 'cmake_build', side_effect=compile_check) as build, \
                 patch.object(setup, 'say'), patch.object(setup, 'ok'), \
                 patch.object(setup, 'source_version', return_value='0.1.30'), \
                 patch.dict(os.environ, {'CCACHE_DISABLE': '0'}):
                setup.build_engine({'arch': 70}, 'none', True, root)
                build.assert_called_once()
                self.assertEqual(os.environ['CCACHE_DISABLE'], '0')
            meta = json.loads((root / 'engine/BUILD.json').read_text())
            self.assertEqual(meta['cpu_signature'], 'new-cpu')
            self.assertEqual((root / 'engine/strata').read_text(), 'rebuilt')

    def test_invalidation_preserves_other_objects_and_build_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cpu = root / 'ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu'
            other = root / 'ggml/src/CMakeFiles/ggml-base.dir'
            cpu.mkdir(parents=True)
            other.mkdir(parents=True)
            files = [cpu / 'quants.c.o', cpu / 'quants.obj', cpu / 'build.make', other / 'base.c.o']
            for file in files:
                file.write_text('fixture')
            setup.invalidate_native_cpu_objects(root)
            self.assertFalse(files[0].exists())
            self.assertFalse(files[1].exists())
            self.assertTrue(files[2].exists())
            self.assertTrue(files[3].exists())


if __name__ == '__main__':
    unittest.main()
