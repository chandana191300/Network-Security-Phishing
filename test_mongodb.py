import os
from pymongo import MongoClient

uri = os.getenv("MONGO_DB_URL")
if not uri:
    raise RuntimeError("Set MONGO_DB_URL before running this MongoDB connection test.")

client = MongoClient(uri, serverSelectionTimeoutMS=5000)
try:
    client.admin.command("ping")
    print("MongoDB connection successful.")
finally:
    client.close()
