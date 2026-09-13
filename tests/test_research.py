import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from kernel_evolution.archive import Archive
from kernel_evolution.research import search


class ResearchTests(unittest.TestCase):
    def test_cached_queries_do_not_make_another_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            archive=Archive(Path(directory)/'archive.sqlite')
            llm=Mock();llm.complete.return_value={'results':[{'url':'https://triton-lang.org/',
                                                             'snippet':'Retrieved reference','code':''}]}
            first=search('Triton API',llm,archive,timeout=10)
            self.assertEqual(first,search('Triton API',llm,archive,timeout=10))
            self.assertEqual(llm.complete.call_count,1)
            self.assertEqual(len(archive.rows('SELECT * FROM search_cache')),1)
            archive.db.close()
