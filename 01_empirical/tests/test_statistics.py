"""Small numerical checks; these fixtures do not represent the study's data."""
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
sys.path.insert(0, str(PACKAGE))
from nev_forecasting.climate import block_counts, holm, load_forecasts
from nev_forecasting.robustness import weighted_auc, attach_actual_exposure
from prepare_market import calculate_technical_indicators


class NumericalTests(unittest.TestCase):
    def test_date_weighted_auc_matches_explicit_duplicates_and_ties(self):
        truth = np.array([0, 1, 1, 0, 1, 0])
        score = np.array([.2, .2, .7, .7, .7, .9])
        dates = np.array([0, 0, 1, 1, 2, 2])
        counts = np.array([[1, 1, 1], [2, 0, 1], [0, 2, 1]])
        actual = weighted_auc(truth, score, dates, counts)
        for k, row in enumerate(counts):
            ix = np.repeat(np.arange(len(truth)), row[dates])
            self.assertAlmostEqual(actual[k], roc_auc_score(truth[ix], score[ix]), places=14)

    def test_bootstrap_preserves_number_of_dates(self):
        counts = block_counts(11, reps=25, block=4, seed=42)
        np.testing.assert_array_equal(counts.sum(axis=1), np.full(26, 11))
        np.testing.assert_array_equal(counts[0], np.ones(11))
        np.testing.assert_array_equal(counts, block_counts(11, reps=25, block=4, seed=42))

    def test_holm_keeps_missing_values_and_input_order(self):
        np.testing.assert_allclose(holm([.04, .001, np.nan, .03]), [.06, .003, np.nan, .06], equal_nan=True)

    def test_exposure_respects_disclosure_date_and_latest_fiscal_year(self):
        dates = pd.to_datetime(['2020-04-01', '2020-04-02', '2020-04-03', '2020-04-04'])
        panel = pd.DataFrame({'stock_code': ['000001']*4, 'datetime': dates, 'window_id': [1]*4,
                              'train_start': ['2019-01-01']*4, 'train_end': ['2020-04-01']*4})
        baseline = panel[['stock_code', 'datetime']].copy()
        events = pd.DataFrame({'stock_code': ['000001']*3, 'report_year': [2018, 2019, 2017],
                               'publication_date': pd.to_datetime(['2019-03-01', '2020-04-02', '2020-04-03']),
                               'available_date': pd.to_datetime(['2019-03-02', '2020-04-03', '2020-04-04']),
                               'exposure': [.1, .2, .9]})
        result, _ = attach_actual_exposure(panel, baseline, events)
        np.testing.assert_allclose(result.exposure, [.1, .1, .2, .2])

    def test_mismatched_frozen_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); predictions = root / 'predictions'; predictions.mkdir()
            for name in ['tech', 'tech_answer', 'tech_question', 'tech_all']:
                frame = pd.DataFrame({'window_id': [1], 'stock_code': ['000001'],
                                      'datetime': [pd.Timestamp('2020-01-01')],
                                      'y_true': [int(name == 'tech_answer')], 'y_prob': [.4], 'y_pred': [0]})
                frame.to_parquet(predictions / f'window_001_XGBoost_{name}.parquet', index=False)
            with self.assertRaisesRegex(ValueError, 'different keys or labels'):
                load_forecasts({'main_interim': root}, 1)

    def test_indicators_do_not_mix_stocks(self):
        dates = pd.date_range('2020-01-01', periods=80)
        frames = []
        for stock, offset in [('000001', 10), ('000002', 100)]:
            close = np.arange(80) + offset
            frames.append(pd.DataFrame({'stock_code': stock, 'datetime': dates, 'open': close,
                                        'close': close, 'high': close+1, 'low': close-1}))
        combined = pd.concat(frames, ignore_index=True)
        actual = calculate_technical_indicators(combined)
        for _, group in actual.groupby('stock_code'):
            np.testing.assert_allclose(group.MA10, group.close.rolling(10).mean(), equal_nan=True)
            np.testing.assert_allclose(group.ROCP, (group.close/group.close.shift(5)-1)*100, equal_nan=True)
            self.assertTrue(group.K.iloc[:8].isna().all())

    def test_daily_alignment_cutoff_and_late_reply_recoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = pd.DataFrame({'stock_code': ['000001'], 'datetime': ['2020-01-03 16:00:00'],
                'question': ['test'], 'question_length': [4], 'q_sentiment_score': [.2], 'q_sentiment_label': [1],
                'whether_answer': [1], 'reply_time': ['2020-01-06 16:00:00'], 'reply': ['reply'],
                'reply_length': [5], 'a_sentiment_score': [.3], 'a_sentiment_label': [1], 'time_diff': [70], 'similarity': [.8]})
            records.to_csv(root/'records.csv', index=False)
            pd.DataFrame({'stock_code': ['000001']*3, 'datetime': ['2020-01-03','2020-01-06','2020-01-07'],
                          'close': [10,11,12]}).to_csv(root/'market.csv',index=False)
            subprocess.run([sys.executable, str(PACKAGE/'prepare_daily.py'), '--interactions', str(root/'records.csv'),
                '--market-features', str(root/'market.csv'), '--output', str(root/'daily.csv')],
                check=True, capture_output=True, text=True)
            result = pd.read_csv(root/'daily.csv').set_index('datetime')
            self.assertEqual(set(result.index), {'2020-01-03','2020-01-06','2020-01-07'})
            self.assertEqual(result.loc['2020-01-06','ques_counts'], .5)
            self.assertEqual(result.loc['2020-01-07','ans_counts'], 0)
            self.assertEqual(result.loc['2020-01-07','ans_time_diff'], 66)
            self.assertEqual(result.loc['2020-01-07','ans_similarity'], 0)


if __name__ == '__main__':
    unittest.main()
