import pymongo
import os


mongodb_uri = os.getenv('MONGODB_URI', default='mongodb://localhost:27017')

db = pymongo.MongoClient(mongodb_uri)["diff_EfficientDet"]

stats = db['inference'].stats.find()
print(stats)
print(stats.next())