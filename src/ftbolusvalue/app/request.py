"""
    Code to send a POST request to the specified URL with the input data as JSON
    in the request body.
    The response content will be printed to the console.
    Make sure your local app server is running and accessible at 'url'
    for this code to work.
"""

# %%
import requests

url = "http://127.0.0.1:8000/api/v1/import_value/predict"
headers = {"Content-Type": "application/json"}

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

response = requests.post(url, headers=headers, json=input_data)

print(response.content)
