import pandas as pd
from sklearn.model_selection import train_test_split
# Notice the new, simplified import paths for version 0.7.x!
from evidently import Report
from evidently.presets import DataDriftPreset

# 1. Load your reference (original training) data
df = pd.read_csv('data/Churn.csv')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 2. Simulate "Production" Data (e.g., incoming data from last week)
# To simulate drift, let's artificially change the Monthly Charges in current data
production_df = test_df.copy()
production_df['Monthly Charge'] = production_df['Monthly Charge'] * 1.5 

# 3. Generate the Data Drift Report using the new syntax
report = Report([
    DataDriftPreset()
])

# Run the report (In the new version, you pass current_data first, then reference_data)
my_eval = report.run(production_df, train_df)

# 4. Save the report as an interactive HTML file
my_eval.save_html("data_drift_report.html")
print("Drift report successfully generated: data_drift_report.html")