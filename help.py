import json
import pymongo

# ✅ Correct MongoDB Atlas connection string
MONGO_URI = "mongodb+srv://ayush:123%40123@cluster0.rrv507u.mongodb.net/?retryWrites=true&w=majority"

# ✅ Load the JSON file
with open(r"C:\Users\AP001118679\Downloads\client_001_002.json", "r") as file:
    data = json.load(file)

# ✅ Fix _id field if it contains $oid
for doc in data:
    if isinstance(doc.get("_id"), dict) and "$oid" in doc["_id"]:
        doc["_id"] = doc["_id"]["$oid"]

# ✅ Connect to MongoDB Atlas
client = pymongo.MongoClient(MONGO_URI)
db = client["wealth_assistant"]  # Replace with your actual DB name
collection = db["clients"]       # Replace with your collection name

# ✅ Insert data
result = collection.insert_many(data)
print(f"Inserted {len(result.inserted_ids)} documents successfully.")
