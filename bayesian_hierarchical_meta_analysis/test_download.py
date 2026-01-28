#!/usr/bin/env python3
"""Test downloading the dataset using kagglesdk directly."""

from kagglesdk.kaggle_client import KaggleClient
from kagglesdk.kaggle_creds import KaggleCredentials
import requests
import zipfile
from pathlib import Path

client = KaggleClient()
creds = KaggleCredentials.load(client)

if not creds:
    print("No credentials found!")
    exit(1)

print(f"Authenticated as: {creds._username}")
print("Access token available:", creds.get_access_token() is not None)

# Try to download using the API directly
dataset_owner = "marian447"
dataset_slug = "retail-scanner-data-gauge-price-elasticity"

# Try different possible dataset identifiers
# The notebook might have the dataset attached with a different name
possible_datasets = [
    f"{dataset_owner}/{dataset_slug}",
    f"{dataset_owner}/retail-scanner-data",  # Without gauge-price-elasticity
    "marian447/retail-scanner",  # Shorter version
]

# Use the kaggle API endpoint
api_url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_owner}/{dataset_slug}"

headers = {
    "Authorization": f"Bearer {creds.get_access_token()}"
}

print(f"\nAttempting to download from: {api_url}")

data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)
zip_path = data_dir / "dataset.zip"

try:
    response = requests.get(api_url, headers=headers, stream=True)
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        print("Downloading...")
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Extracting to {data_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Find CSV files
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            if len(csv_files) == 1:
                output_file = data_dir / "scanner_data.csv"
                csv_files[0].rename(output_file)
                print(f"Renamed to: {output_file}")
            else:
                print(f"Found multiple CSV files: {[f.name for f in csv_files]}")
        
        zip_path.unlink()  # Remove zip file
        print("Download complete!")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
