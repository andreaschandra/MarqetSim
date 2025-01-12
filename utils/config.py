import os
from dotenv import load_dotenv


class Config:
    def __init__(self, logger):
        if os.path.exists(".env.local"):
            load_dotenv(".env.local")
            print("Using .env.local")
        else:
            load_dotenv()
            print("Using .env")
