"""
Example DBRepo REST API loader.
"""

import requests
import pandas as pd

DBREPO_DATABASE_URL = "https://test.dbrepo.tuwien.ac.at/database/ed511834-2154-4cee-8676-0ec57829e465/info"
DBREPO_VIEW_PID = "https://handle.test.datacite.org/10.82556/037y-wk92"
DBREPO_SUBSET_PID = "https://handle.test.datacite.org/10.82556/8xrb-a603"

def load_from_api(api_endpoint: str) -> pd.DataFrame:
    try:
        response = requests.get(api_endpoint, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Could not retrieve DBRepo data: {exc}") from exc

    payload = response.json()
    rows = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print("DBRepo database:", DBREPO_DATABASE_URL)
    print("DBRepo view PID:", DBREPO_VIEW_PID)
    print("DBRepo subset PID:", DBREPO_SUBSET_PID)
