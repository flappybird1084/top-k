from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock
from kernelevo.dispatch_once import run_once
class Once(TestCase):
 def test_duplicate_queue_entry_does_not_relaunch(self):
  with TemporaryDirectory() as root:
   launch=Mock(return_value=0)
   for _ in range(2):self.assertEqual(run_once(Path(root)/'job.json',launch),0)
   launch.assert_called_once()
 def test_unknown_prior_launch_is_not_repeated(self):
  with TemporaryDirectory() as root:
   (Path(root)/'dispatch-started').touch();launch=Mock()
   with self.assertRaises(RuntimeError):run_once(Path(root)/'job.json',launch)
   launch.assert_not_called()
