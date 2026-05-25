import json
import pandas as pd
from google.cloud.firestore import Client
from google.oauth2 import service_account

with open(r'c:\userdata\python\NSEStockDataAnalysis\.streamlit\secrets.toml', 'r') as f:
    lines = f.readlines()

json_lines = []
for line in lines:
    stripped = line.strip()
    if stripped.startswith('#') or stripped.startswith('[') or not stripped:
        continue
    json_lines.append(stripped)

json_str = '{' + '\n'.join(json_lines) + '}'
json_str = json_str.replace(',\n}', '\n}')
cred_info = json.loads(json_str)

credentials = service_account.Credentials.from_service_account_info(cred_info)
db = Client(credentials=credentials, project=cred_info['project_id'], database='db-analytics')

# Fetch a few records just to see
docs = list(db.collection('stock_metrics').limit(5).stream())
data = []
for doc in docs:
    data.append(doc.to_dict())

df = pd.DataFrame(data)
print(df.head())
print("Columns:", df.columns.tolist())
