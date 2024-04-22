import Signals_Extraction as se
import Preprocess as pp

se.main("trial_lie_001.mp4", "Test.csv")

#signals_df = pp.load_data_from_csv("temp.csv")
#features_df = pp.extract_features(signals_df)

import pandas as pd
#features_df.to_csv("trial_lie_001.csv", index= False)