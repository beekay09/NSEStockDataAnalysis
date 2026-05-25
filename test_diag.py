try:
    import json
    import pandas as pd
    import App
    from google.cloud.firestore import Client
    from google.oauth2 import service_account

    with open(r".streamlit\secrets.toml", "r") as f:
        lines = f.readlines()
        
    json_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("[") or not stripped:
            continue
        json_lines.append(stripped)
        
    json_str = "{" + "\n".join(json_lines) + "}"
    json_str = json_str.replace(",\n}", "\n}")
    cred_info = json.loads(json_str)

    credentials = service_account.Credentials.from_service_account_info(cred_info)
    db = Client(credentials=credentials, project=cred_info['project_id'], database='db-analytics')
    print("Fetching data...")
    docs = db.collection('stock_metrics').stream()
    data = [doc.to_dict() for doc in docs]
    print(f"Data length: {len(data)}")
    df = pd.DataFrame(data)
    print("DataFrame shape:", df.shape)
except Exception as e:
    print(f"Error: {e}")
