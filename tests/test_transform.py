import pandas as pd
import unittest
import pyterrier as pt
pt.init()
import pyterrier_doc2query

class TestTransform(unittest.TestCase):

    def test_transform(self):
        doc2query = pyterrier_doc2query.Doc2Query()
        input = pd.DataFrame([
            {'docno': '1', 'text': 'Hello Terrier!'}
        ])
        out = doc2query(input)
        self.assertTrue('querygen' in out.columns)
        self.assertTrue('terrier' in out['querygen'][0].lower())

        # append mode
        doc2query.append = True
        out = doc2query(input)
        self.assertFalse('querygen' in out.columns)
        self.assertTrue(out['text'][0].startswith('Hello Terrier!'))
