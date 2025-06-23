from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

client = AsyncIOMotorClient(settings.MONGODB_URL)
database = client[settings.DATABASE_NAME]
user_collection = database.get_collection("users")
paper_collection = database.get_collection("papers")
chat_collection = database.get_collection("chats")
