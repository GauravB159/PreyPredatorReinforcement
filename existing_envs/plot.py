import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('existing_envs/log_test_20240404-210735.csv')

print(data.columns)
# Assume the CSV file has two columns 'x' and 'y'
plt.plot(data['Episode'], data['Test_Return'])

# Show the plot
plt.show()