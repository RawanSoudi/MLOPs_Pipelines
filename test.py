import dagshub
import os

print(f"Token: {os.getenv('DAGSHUB_TOKEN')}")
print(f"Repo: {os.getenv('DAGSHUB_REPO')}")
print(f"User: {os.getenv('DAGSHUB_USERNAME')}")