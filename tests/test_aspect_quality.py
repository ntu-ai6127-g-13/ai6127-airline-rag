import unittest
from experiments.aspect_quality.prompts import majority_aspects,system_prompt,ASPECTS


class AspectQualityTests(unittest.TestCase):
    def test_majority_does_not_override_three_votes_with_fewshot(self):
        self.assertEqual(majority_aspects([['seat'],['seat'],['food'],['seat']]),['seat'])

    def test_tie_prefers_fewshot_only_when_it_is_among_top_votes(self):
        self.assertEqual(majority_aspects([['seat'],['food'],['food'],['seat']]),['food'])
        self.assertEqual(majority_aspects([['seat'],['seat'],['food'],['other']]),['seat'])

    def test_rotated_prompt_relabels_examples_consistently(self):
        prompt,order=system_prompt('rotated',[{'text':'Sample sentence.','aspect':'seat'}])
        self.assertEqual(set(order),set(ASPECTS))
        letter=chr(65+order.index('seat'))
        self.assertIn(letter+': passenger seat comfort',prompt)
        self.assertIn('Answer: '+letter,prompt)

    def test_incomplete_ensemble_is_rejected(self):
        with self.assertRaises(ValueError):majority_aspects([['seat'],[],['food'],['seat']])
