import pandas as pd
import unittest
import pyterrier_doc2query
import pyterrier_alpha as pta

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

    def test_empty_input(self):
        doc2query = pyterrier_doc2query.Doc2Query()
        res = doc2query(pd.DataFrame([], columns=['docno', 'text', 'something_else']))
        self.assertEqual(list(res.columns), ['docno', 'text', 'something_else', 'querygen'])
        doc2query.append = True
        res = doc2query(pd.DataFrame([], columns=['docno', 'text', 'something_else']))
        self.assertEqual(list(res.columns), ['docno', 'text', 'something_else'])
        with self.assertRaises(pta.validate.InputValidationError):
            doc2query(pd.DataFrame([], columns=['docno', 'something_else'])) # missing text column
