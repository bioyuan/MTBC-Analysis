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

    def test_regions_derived_from_province(self):
        provinces = [
            " \u9ed1\u9f99\u6c5f\u7701 ", "beijing", "Shaanxi",
            "\u4e0a\u6d77\u5e02", "Guangxi", "Tibet",
        ]
        rng = np.random.default_rng(42)
        n = 90 * len(provinces)
        frame = pd.DataFrame({
            "province": np.repeat(provinces, 90),
            "age": rng.integers(1, 90, n),
            "sex": rng.choice(["Female", "Male"], n),
            "mtbc": rng.integers(0, 2, n),
            "outcome": rng.integers(0, 2, n),
        })
        expected = {"North": 270, "East": 90, "Central-Southern": 90, "Southwest": 90}
        for source in (frame, frame.assign(region="outdated")):
            results = m.run(source, ["outcome"]).query("model == 'region'")
            self.assertEqual(results.set_index("group")["n"].to_dict(), expected)
            self.assertEqual(set(results["status"]), {"ok"})

    def test_missing_and_unrecognized_provinces(self):
        self.assertTrue(m.derive_region(pd.Series([None, " "])).isna().all())
        with self.assertRaisesRegex(ValueError, "Unrecognized province labels"):
            m.derive_region(pd.Series(["Unknown province"]))

if __name__ == "__main__":unittest.main()
