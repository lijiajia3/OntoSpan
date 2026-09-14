"""Public launcher boundaries; subprocesses mocked, no installs or GPU work."""
import importlib.util
import builtins
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent


class LauncherTests(unittest.TestCase):
    def test_dispatch_preserves_arguments_exit_code_and_stops_recursion(self):
        import windows_launcher as launcher
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / '.venv' / 'Scripts' / 'python.exe'
            target.parent.mkdir(parents=True)
            target.touch()
            script = root / '07_SUMMARIZE.py'
            with patch.object(launcher, 'ROOT', root), patch.object(launcher.platform, 'system', return_value='Windows'), \
                 patch.object(sys, 'argv', [str(script), '--output', 'outputs/中文 路径']), \
                 patch.object(launcher.subprocess, 'call', return_value=7) as call:
                with self.assertRaises(SystemExit) as exit_result:
                    launcher.ensure_runtime()
                self.assertEqual(exit_result.exception.code, 7)
                self.assertEqual(call.call_args.args[0], [str(target), str(script.resolve()), '--output', 'outputs/中文 路径'])
                self.assertEqual(call.call_args.kwargs['env']['PYTHONIOENCODING'], 'utf-8')
                call.reset_mock()
                with patch.object(sys, 'executable', str(target)), patch.object(sys, 'version_info', (3, 11, 9)):
                    launcher.ensure_runtime()
                call.assert_not_called()

    def test_missing_venv_stops_with_setup_instruction(self):
        import windows_launcher as launcher
        with tempfile.TemporaryDirectory() as tmp, patch.object(launcher, 'ROOT', Path(tmp)), \
             patch.object(launcher.platform, 'system', return_value='Windows'), \
             patch.object(launcher.subprocess, 'call') as call:
            with self.assertRaisesRegex(SystemExit, '01_SETUP_WINDOWS.py'):
                launcher.ensure_runtime()
            call.assert_not_called()

    def test_setup_failure_stops_without_check_or_training(self):
        import windows_launcher as launcher
        with tempfile.TemporaryDirectory() as tmp, patch.object(launcher, 'ROOT', Path(tmp)), \
             patch.object(launcher.platform, 'system', return_value='Windows'), \
             patch.object(sys, 'version_info', (3, 11, 9)), \
             patch.object(launcher.subprocess, 'run', side_effect=launcher.subprocess.CalledProcessError(1, ['venv'])) as run:
            with self.assertRaises(launcher.subprocess.CalledProcessError):
                launcher.setup()
            self.assertEqual(run.call_count, 1)

    def test_existing_venv_reused_and_offline_models_respected(self):
        import windows_launcher as launcher
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / '.venv' / 'Scripts' / 'python.exe'
            target.parent.mkdir(parents=True)
            target.touch()
            with patch.object(launcher, 'ROOT', root), patch.object(launcher.platform, 'system', return_value='Windows'), \
                 patch.object(sys, 'version_info', (3, 11, 9)), patch.object(launcher.subprocess, 'run') as run:
                launcher.setup(download=False)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertTrue(all(cmd[0] == str(target) for cmd in commands))
            self.assertFalse(any('venv' in cmd for cmd in commands))
            self.assertIn('preflight(download=False)', commands[-1][-1])

    def test_all_entries_switch_before_loading_training_dependencies(self):
        import windows_launcher as launcher
        real_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name in {'torch', 'runner', 'report', 'protocol'}:
                raise AssertionError('Dependencies loaded before environment switch: ' + name)
            return real_import(name, *args, **kwargs)
        for entry in sorted(ROOT.glob('0[2-8]_*.py')):
            with self.subTest(entry=entry.name), \
                 patch.object(launcher, 'ensure_runtime', side_effect=RuntimeError('runtime selected')), \
                 patch.object(launcher.subprocess, 'call', side_effect=AssertionError('Child started before environment switch')), \
                 patch.object(builtins, '__import__', side_effect=guarded_import):
                with self.assertRaisesRegex(RuntimeError, 'runtime selected'):
                    runpy.run_path(str(entry), run_name='__main__')

    def test_setup_installs_in_local_venv_without_importing_torch(self):
        spec = importlib.util.spec_from_file_location('setup_entry', ROOT / '01_SETUP_WINDOWS.py')
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        with tempfile.TemporaryDirectory() as tmp:
            import windows_launcher as launcher
            root = Path(tmp)
            with patch.object(launcher, 'ROOT', root), patch.object(launcher.platform, 'system', return_value='Windows'), \
                 patch.object(sys, 'version_info', (3, 11, 9)), patch.object(sys, 'argv', ['01_SETUP_WINDOWS.py']), \
                 patch.object(launcher.subprocess, 'run') as run:
                entry.main()
            commands = [call.args[0] for call in run.call_args_list]
            target = str(root / '.venv' / 'Scripts' / 'python.exe')
            self.assertEqual(commands[0], [sys.executable, '-m', 'venv', str(root / '.venv')])
            self.assertTrue(all(cmd[0] == target for cmd in commands[1:]))
            self.assertTrue(any('torch==2.9.1' in cmd for cmd in commands))
            self.assertIn('preflight(download=True)', commands[-1][-1])


if __name__ == '__main__':
    unittest.main()
