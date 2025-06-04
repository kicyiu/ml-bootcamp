from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import io
import requests

data_url = "http://lib.stat.cmu.edu/datasets/boston"
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

try:
    # Fetch the raw text content
    response = requests.get(data_url)
    response.raise_for_status() # Raise an exception for bad status codes
    raw_content = response.text
except requests.exceptions.RequestException as e:
    print(f"Error fetching data from {data_url}: {e}")
    print("Please check your internet connection or if the URL is still valid.")
    exit()

# Split the content into lines
lines = raw_content.split('\n')

# The actual data starts after a preamble (usually line 22, 0-indexed)
# Let's find the start of the data by looking for lines that are not part of the description
data_start_line = 0
for i, line in enumerate(lines):
    if line.strip().startswith('1.'): # Heuristic: data usually starts with "1." for the first row
        data_start_line = i
        break
    elif i > 25: # Fallback: if not found by line 25, assume data starts after standard preamble
        data_start_line = 22
        break
# If the loop finishes without finding "1.", data_start_line will be 0.
# A more robust check might be to look for empty lines after the description.

# Process the lines to combine two physical lines into one logical row
processed_lines = []
temp_row = []
for i in range(data_start_line, len(lines)):
    line = lines[i].strip()
    if not line: # Skip empty lines
        continue

    # Split the numbers by one or more spaces
    current_numbers = [x for x in line.split(' ') if x]

    # Extend the current_row being built
    temp_row.extend(current_numbers)

    # Each logical row has 14 values (13 features + 1 target)
    if len(temp_row) == 14:
        processed_lines.append(" ".join(temp_row)) # Join numbers with space for pd.read_csv
        temp_row = [] # Reset for the next logical row
    elif len(temp_row) > 14: # Handle unexpected extra values if any
        print(f"Warning: Row has more than 14 values. Skipping: {temp_row}")
        temp_row = [] # Reset to avoid cascading errors

# Create a StringIO object from the processed lines
# This simulates a file-like object that pandas can read
data_io = io.StringIO("\n".join(processed_lines))

# Now, read this into a DataFrame
# Use sep=r"\s+" because the numbers are space-separated
# No need for skiprows or header=None since we've pre-processed
data_frame = pd.read_csv(data_io, sep=r"\s+", header=None, names=column_names)


features = data_frame.drop(['INDUS', 'AGE', 'MEDV'], axis=1)
target_values = data_frame['MEDV']

log_prices = np.log(target_values).values
target = pd.DataFrame(log_prices, columns=['PRICE'])

CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8

ZILLOW_MEDIA_PRICE = 807.9 # ZILLOW_MEDIA_PRICE for boston house values in Jun 2025
SCALE_FACTOR = ZILLOW_MEDIA_PRICE / np.median(target_values)

property_stats = features.mean().values.reshape(1, 11)

regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)


MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)


def get_log_estimate(nr_rooms,
                     students_per_classroom,
                     next_to_river=False,
                     high_confidence=True):
    
    # Configure property
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classroom

    if next_to_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0

    # Make prediction
    log_estimate = regr.predict(property_stats)[0][0]

    # Calc Range
    if high_confidence:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68

    return log_estimate, upper_bound, lower_bound, interval


def get_dollar_estimate(rm, ptratio, chas=False, large_range=True):
    """Estimate the price of a property in Boston.

    Keywords arguments:
    rm -- number of rooms in the property.
    ptratio -- number of students per teacher in the classroom for the school in the area.
    chas -- True if the property is next to the river, False otherwhise.
    large_range -- True for a 95% prediction interval, False for a 68% interval.
    
    """

    if rm < 1 or ptratio < 1:
        print('That is unrealistic. Try again.')
        return

    log_est, upper, lower, conf = get_log_estimate(rm,
                                                   students_per_classroom=ptratio,
                                                   next_to_river=chas,
                                                   high_confidence=large_range)

    # Convert to today's dollars
    dollar_est = np.e**log_est * 1000 * SCALE_FACTOR
    dollar_hi = np.e**upper * 1000 * SCALE_FACTOR
    dollar_low = np.e**lower * 1000 * SCALE_FACTOR
    
    # Round the dollar values to nearest thousand
    rounded_east = np.around(dollar_est, -3)
    rounded_hi = np.around(dollar_hi, -3)
    rounded_low = np.around(dollar_low, -3)
    
    print(f'The estimated property values is {rounded_east}')
    print(f'At {conf}% confidence the valuation range is')
    print(f'USD {rounded_low} at the lower to USD {rounded_hi} at the high end.')

