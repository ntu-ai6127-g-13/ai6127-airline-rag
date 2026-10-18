import unittest
import pandas as pd
from src.weak_label_rules import match_series
from src.weak_labels import build_weak_labels
from tests.test_weak_labels import row


def detected(text):
    m, blocked = match_series(pd.Series([text]))
    return {a for a in m if m.at[0, a]}, bool(blocked.iloc[0])


class ContextRuleTests(unittest.TestCase):
    def test_plural_and_hyphen_variants(self):
        for text, aspect in [('Seats have ample leg-room.', 'seat'),
                             ('The leg–room is generous.', 'seat'),
                             ('Meals and beverages arrived.', 'food'),
                             ('Flight attendants were helpful.', 'staff')]:
            with self.subTest(text=text):
                self.assertEqual(detected(text)[0], {aspect})

    def test_screen_requires_entertainment_context(self):
        for text in ['The website screen froze during payment.',
                     'She did not look at her screen.',
                     'They damaged the TV in my luggage.']:
            with self.subTest(text=text):
                self.assertNotIn('entertainment', detected(text)[0])
        self.assertIn('entertainment', detected('The seatback screen offered films.')[0])
        self.assertIn('entertainment', detected('The TV was in the armrest.')[0])

    def test_staff_is_not_any_employee(self):
        self.assertEqual(detected('Airport staff at the counter were rude.')[0], {'ground'})
        self.assertEqual(detected('Customer service refused the refund.')[0], {'value'})
        self.assertEqual(detected('Onboard staff were very friendly.')[0], {'staff'})
        self.assertEqual(detected('The Cabin Manager was attentive.')[0], {'staff'})
        self.assertEqual(detected('The service was good.')[0], set())

    def test_ambiguous_comfort_space_and_ground(self):
        self.assertNotIn('seat', detected('We took a bus from a remote parking space.')[0])
        self.assertEqual(detected('Delays are uncomfortable.')[0], {'ground'})
        self.assertEqual(detected('The lounge is on the ground floor.')[0], set())
        self.assertIn('seat', detected('The seat was uncomfortable.')[0])

    def test_background_and_multi_aspect_abstain(self):
        self.assertEqual(detected('Meals ran out, merely for cost saving.')[0], {'food'})
        self.assertEqual(detected('Check-in was slow but seats had leg-room.')[0], {'ground', 'seat'})
        self.assertEqual(detected('Flight attendants ignored passengers asking for drinks.')[0], {'staff', 'food'})
        self.assertTrue(detected('The plane was dirty and food was stuck to the seats.')[1])

    def test_new_filter_and_original_text_preservation(self):
        source='The seats have good leg–room.'
        out, report=build_weak_labels(pd.DataFrame([
            row('keep',source),row('multi','Meals were good but flight attendants were rude.'),
            row('dirty','The cabin was dirty and food was stuck to seats.')]),set(),set())
        self.assertEqual(out.chunk_id.tolist(), ['keep'])
        self.assertEqual(out.text.iloc[0],source)
        self.assertEqual(report['excluded_by_first_reason']['multiple_aspects'],1)
        self.assertEqual(report['excluded_by_first_reason']['unsupported_aspect'],1)
        self.assertTrue(all(report['checks'].values()))

    def test_legacy_remains_available(self):
        sample=pd.DataFrame([row('old','The ground floor has chairs.'),row('new','The meals were good.')])
        out,_=build_weak_labels(sample,set(),set(),ruleset='legacy')
        self.assertEqual(out.chunk_id.tolist(),['old'])
        out,_=build_weak_labels(sample,set(),set())
        self.assertEqual(out.chunk_id.tolist(),['new'])


if __name__=='__main__': unittest.main()
