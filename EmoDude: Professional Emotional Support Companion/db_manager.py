# utils/db_manager.py
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

load_dotenv()
client = MongoClient(os.getenv("MONGODB_URI"))
db = client.emodude
chats = db.chats

def save_chat_history(session_id: str, user_input: str, response: str, emotions: dict):
    try:
        chats.update_one(
            {"session_id": session_id},
            {"$push": {"history": {"user_input": user_input, "response": response, "emotions": emotions}}},
            upsert=True
        )
    except Exception as e:
        logger.error(f"DB save error: {e}")

def get_chat_history(session_id: str) -> list:
    try:
        doc = chats.find_one({"session_id": session_id})
        return doc.get("history", []) if doc else []
    except Exception as e:
        logger.error(f"DB fetch error: {e}")
        return []