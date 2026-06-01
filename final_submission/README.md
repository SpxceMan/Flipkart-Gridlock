# Solution Summary

- Problem description: Predict demand accurately based on spatiotemporal features like geohash, weather, timestamp, and road type data in a regression hackathon format.
- Setup instructions: Use Python 3.9+ and run `pip install -r requirements.txt` to install the necessary dependencies.
- How to run solution.py: Simply execute `python solution.py` in the same directory as `train.csv` and `test.csv`. The script will automatically parse features, run 5-fold cross-validation to train the models, apply Optuna tuning, and optimize ensemble weights. 
- How submission.csv is generated: After running the script, `submission.csv` is automatically created in the current directory with the test predictions based on a weighted ensemble of LightGBM, XGBoost, CatBoost, ExtraTrees, and RandomForest averaged from CV folds and full-retrained models.
