from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != "11221122":  # Replace with a secure method
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key_header