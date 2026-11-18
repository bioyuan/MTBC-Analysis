import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
spec=importlib.util.spec_from_file_location("model",Path(__file__).resolve().parents[1]/"02_code/01_logistic_regression.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ModelTests(unittest.TestCase):
    def test_crude_odds_ratio(self):
        d=pd.DataFrame({"mtbc":[0]*100+[1]*100,"outcome":[1]*20+[0]*80+[1]*40+[0]*60})
        r=m.fit_model(d,"outcome","crude")
        self.assertEqual(r["status"],"ok");self.assertAlmostEqual(r["or"],(40*80)/(60*20),places=5)
    def test_small_stratum_excluded(self):
        d=pd.DataFrame({"mtbc":[0,1]*10,"outcome":[1]*5+[0]*15,"age":range(20),"province":["X"]*20})
        self.assertEqual(m.fit_model(d,"outcome","sex")["status"],"excluded_positive_count_le_5")
    def test_covariates(self):
        self.assertEqual(m.COVARIATES["crude"],[])
        self.assertEqual(m.COVARIATES["adjusted"],["age","sex","province"])
        self.assertEqual(m.COVARIATES["sex"],["age","province"])
        self.assertEqual(m.COVARIATES["age"],["sex","province"])
        self.assertEqual(m.COVARIATES["region"],["age","sex"])
if __name__ == "__main__":unittest.main()
