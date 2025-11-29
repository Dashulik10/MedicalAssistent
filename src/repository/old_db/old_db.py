from pymongo import MongoClient

from configuration.settings import settings

client = MongoClient(settings.mongo_url)
database = client[settings.mongo_initdb_database]
collection = database[settings.mongo_collection]