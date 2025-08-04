from pymongo import MongoClient
from config import settings
 
class FAQStore:
    def __init__(self, db_name):
        client=MongoClient(settings.MONGO_URI)
        db=client[db_name]
        self.collection=db["faqs"]
   
    def get_all_faqs(self):
        return list(self.collection.find({}, {"_id":0, "question":1}))
 