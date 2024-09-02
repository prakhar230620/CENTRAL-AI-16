# backend/config/settings.py

import os
from pydantic import BaseSettings, Field, ValidationError, validator
from typing import Optional

class Settings(BaseSettings):
    ENVIRONMENT: str = Field(..., env='ENVIRONMENT')
    DATABASE_URL: str = Field(..., env='DATABASE_URL')
    API_KEY: str = Field(..., env='API_KEY')
    SERVICE_ENDPOINT: str = Field(..., env='SERVICE_ENDPOINT')

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

    @validator('ENVIRONMENT')
    def validate_environment(cls, v):
        if v not in ['development', 'testing', 'production']:
            raise ValueError("Environment must be 'development', 'testing', or 'production'")
        return v

def get_settings():
    try:
        settings = Settings()
        return settings
    except ValidationError as e:
        print(f"Configuration error: {e}")
        raise

# Example usage:
# settings = get_settings()
# print(settings.DATABASE_URL)
# print(settings.API_KEY)
# print(settings.SERVICE_ENDPOINT)