# Water Tank Monitoring using Linear Regression

## Project overview
This beginner-friendly machine learning project predicts remaining water in a tank (liters) using:
- Time (converted to elapsed minutes)
- Water level (%)
- Water usage rate (liters/minute)

The model is implemented in a Google Colab notebook using Python, pandas, matplotlib, and scikit-learn.

## Files
- `Water_Tank_Linear_Regression.ipynb` — notebook to run in Google Colab
- `water_tank_dataset_time_level_usage.csv` — sample dataset
- `requirements.txt` — Python dependencies

## How to run in Google Colab
1. Download or clone this repository.
2. Open [Google Colab](https://colab.research.google.com/).
3. Choose **File → Upload notebook** and upload `Water_Tank_Linear_Regression.ipynb`.
4. Run the notebook cells in order.
5. When prompted, upload `water_tank_dataset_time_level_usage.csv`.
6. Review the MAE, RMSE, R² score, and actual-vs-predicted graph.

## Dataset columns
| Column | Meaning |
|---|---|
| `time` | Timestamp of the reading |
| `water_level_percent` | Water level as a percentage |
| `usage_rate_liters_per_min` | Estimated water usage rate |
| `remaining_water_liters` | Target value to predict |

## Important limitation
The included dataset is **synthetic**, not collected from a real tank. It assumes a 1000-liter tank and calculates remaining water as water-level percentage × 10. Consequently, water level is directly related to the target and may make the evaluation look unusually good. For a meaningful real-world model, collect real sensor data and independently measured remaining-water values.

## Model evaluation
The notebook reports:
- **MAE** — mean absolute prediction error in liters
- **RMSE** — root mean squared prediction error in liters
- **R²** — proportion of target variation explained by the model

## Technologies
Python · pandas · NumPy · Matplotlib · scikit-learn · Google Colab · GitHub
