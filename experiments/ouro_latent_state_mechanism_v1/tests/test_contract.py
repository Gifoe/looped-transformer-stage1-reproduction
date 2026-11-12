"""Fast CPU-only format tests, not performance/model parity tests."""
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import question_prompt,demo_prefix,make_requests,sha256_text

class TestJointContract(unittest.TestCase):
    def setUp(self):
        self.row=dict(id='arc_challenge/test/x',split='test',question='What is two plus two?',
                      choices=['3','4','5','7'])

    def test_prompt(self):
        s=question_prompt(self.row['question'],self.row['choices'])
        self.assertEqual(s,'Question: What is two plus two?\nA. 3\nB. 4\nC. 5\nD. 7\nAnswer:')

    def test_no_gold(self):
        prompt,requests=make_requests(self.row,'DEMONSTRATIONS\n\n',[0,1,2,3])
        self.assertNotIn('gold',prompt)
        self.assertEqual([q[1] for q in requests],[' A',' B',' C',' D'])
        self.assertEqual(len({q[0] for q in requests}),1)

    def test_rotation(self):
        prompt,requests=make_requests(self.row,'D\n\n',[1,2,3,0])
        self.assertIn('A. 4\nB. 5\nC. 7\nD. 3',prompt)

    def test_25shot(self):
        docs=[dict(id=str(i),question='Q?',choices=dict(text=['a','b'],label=['A','B']),answerKey='B') for i in range(25)]
        d=dict(ids=[str(i) for i in range(25)],documents=docs)
        s=demo_prefix(d)
        self.assertEqual(s.count('Question: Q?'),25)
        self.assertEqual(s.count('Answer: B'),25)
        self.assertTrue(s.endswith('\n\n'))

if __name__=='__main__':unittest.main()
