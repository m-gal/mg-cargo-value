"""
Testing the deployed model and monitoring its performance are crucial
for ensuring reliability.
"""

# %%
import requests
import json
import time
from tqdm import tqdm

NUM_REQUESTS = 100


# Function to send test requests to the deployed model
def test_model(endpoint, input_data):
    headers = {"Content-Type": "application/json"}
    response = requests.post(endpoint, headers=headers, json=input_data)
    return response.json()


# Test the model with sample input data
input_data = [
    {
        "ade_month": "OCT",
        "hscode_04": "3923",
        "weight": 6240,
        "description": "SPOKE CORE PKGS X X MM SPOKES HS CODE SCAC BANQ HBL SEL FREIGHT COLLECT",
    },
    {
        "ade_month": "APR",
        "hscode_04": "6403",
        "weight": 1083,
        "description": "SPORT SHOES FOOTWEAR LEATHER UPPER MENS CTNS PAI RSGROSS PO NO CUSTOMER ORDER NO EC ARTICLE NO B SIZE QUANTITY HS CODE M ADE IN INDONESIA INV NO NG ",
    },
    {
        "ade_month": "FEB",
        "hscode_04": "6105",
        "weight": 919,
        "description": "MEN S RECYCLE POLYESTER KNITTED SWEATSHIRT S C NO HS CODE PO NO CUST NO QTY ",
    },
]

"""
True Values:
    "value": 1240,
    "value": 9747,
    "value": 8271,
"""

endpoint = "http://127.0.0.1:8000/api/v1/import_value/predict"

# Test the model and measure response time
start_time = time.time()
response = test_model(endpoint, input_data)
end_time = time.time()

# Print the response and response time
print("Response:", response)
print("Response Time:", end_time - start_time, "seconds")

# Perform load testing to measure performance under high load
start_time = time.time()

for _ in tqdm(
    range(NUM_REQUESTS),
    desc="Perform testing",
    bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}",
):
    response = test_model(endpoint, input_data)

end_time = time.time()
total_time = end_time - start_time
average_response_time = total_time / NUM_REQUESTS

# Print load testing results
print("Total Requests:", NUM_REQUESTS)
print("Total Time:", total_time, "seconds")
print("Average Response Time:", average_response_time, "seconds")
