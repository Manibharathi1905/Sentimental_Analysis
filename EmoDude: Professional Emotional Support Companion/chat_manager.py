# utils/chat_manager.py
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from config import storage_config
# Defer importing sentence-transformers until runtime to avoid importing
# heavy ML dependencies (torch/torchvision) during module import.
SentenceTransformer = None
from sklearn.metrics.pairwise import cosine_similarity
import logging
import numpy as np
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class ChatManager:
    _lock = threading.Lock()  # For thread-safe SQLite operations

    def __init__(self):
        self.db_path = storage_config.db_path
        self.create_tables()
        self.user_profiles = {}  # In-memory user profiles for adaptive learning
        self._init_semantic_memory()

    def _init_semantic_memory(self):
        """Initialize semantic memory with lazy loading"""
        global SentenceTransformer
        if SentenceTransformer is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                logger.warning("Sentence-Transformers not available; semantic search disabled")
                SentenceTransformer = None

    def create_tables(self):
        """Enhanced table creation with indexes for performance"""
        try:
            with self._lock, sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Sessions table - Enhanced with profile fields
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at DATETIME,
                        last_active DATETIME,
                        user_profile JSON,
                        emotional_signature TEXT,
                        therapy_preference TEXT,
                        total_interactions INTEGER DEFAULT 0
                    )
                """)
                # Messages table - Enhanced with metadata
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp DATETIME,
                        role TEXT NOT NULL,
                        content TEXT,
                        emotions JSON,
                        learned_insights JSON,
                        response_rating REAL,  -- Future user feedback
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                """)
                # Learned patterns - Enhanced for ML adaptation
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS learned_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        pattern_type TEXT,
                        input_pattern TEXT,
                        emotional_response JSON,
                        therapeutic_strategy TEXT,
                        effectiveness REAL,
                        created_at DATETIME,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                """)
                # Add indexes for performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_session ON learned_patterns(session_id)")
                conn.commit()
                logger.info("Database tables created/verified with indexes")
        except sqlite3.Error as e:
            logger.error(f"Database creation failed: {e}")

    def add_exchange(self, session_id: str, user_input: str, bot_response: str,
                     emotions: Dict, learned_insights: Optional[Dict] = None):
        """Add user-bot exchange and update session emotional signature - Enhanced with activity tracking"""
        try:
            with self._lock, sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()
                timestamp = datetime.now()

                # Insert user message
                cursor.execute("""
                    INSERT INTO messages (session_id, timestamp, role, content, emotions, learned_insights)
                    VALUES (?, ?, 'user', ?, ?, ?)
                """, (session_id, timestamp, user_input, json.dumps(emotions), json.dumps(learned_insights or {})))

                # Insert bot response
                cursor.execute("""
                    INSERT INTO messages (session_id, timestamp, role, content, emotions, learned_insights)
                    VALUES (?, ?, 'assistant', ?, ?, ?)
                """, (session_id, timestamp, bot_response, json.dumps(emotions), json.dumps(learned_insights or {})))

                # Store learned patterns if present - Enhanced validation
                if learned_insights and isinstance(learned_insights, dict):
                    cursor.execute("""
                        INSERT INTO learned_patterns (session_id, pattern_type, input_pattern, emotional_response, therapeutic_strategy, effectiveness, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        session_id,
                        learned_insights.get('type', 'general'),
                        user_input[:100],  # Truncate for DB efficiency
                        json.dumps(emotions),
                        learned_insights.get('strategy', ''),
                        learned_insights.get('effectiveness', 0.0),
                        timestamp
                    ))

                # Update session emotional signature and activity
                recent_emotions = self.get_recent_emotions(session_id, limit=10)
                emotional_signature = self._calculate_emotional_signature(recent_emotions)
                total_interactions = self.get_total_interactions(session_id) + 2  # User + bot
                cursor.execute("""
                    INSERT OR REPLACE INTO sessions (session_id, created_at, last_active, emotional_signature, total_interactions)
                    VALUES (?, COALESCE((SELECT created_at FROM sessions WHERE session_id=?), ?), ?, ?, ?)
                """, (session_id, session_id, timestamp, timestamp, emotional_signature, total_interactions))
                conn.commit()
                logger.debug(f"Exchange added for session {session_id[:8]}...")
        except sqlite3.Error as e:
            logger.error(f"Error in add_exchange: {e}")

    def get_messages(self, session_id: str, limit: int = 100) -> List[Dict]:
        """Retrieve messages with enhanced metadata"""
        try:
            with self._lock, sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT role, content, emotions, learned_insights, timestamp, response_rating 
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                """, (session_id, limit))
                rows = cursor.fetchall()
                messages = []
                for row in rows:
                    msg = {
                        "role": row[0],
                        "content": row[1] or "",
                        "emotions": json.loads(row[2]) if row[2] else {},
                        "insights": json.loads(row[3]) if row[3] else {},
                        "timestamp": row[4],
                        "rating": row[5]  # Future feedback
                    }
                    # Add derived fields
                    if msg["emotions"]:
                        msg["primary_emotion"] = max(msg["emotions"], key=msg["emotions"].get)
                    messages.append(msg)
                return messages
        except sqlite3.Error as e:
            logger.error(f"Error in get_messages: {e}")
            return []

    def get_recent_emotions(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Return recent user emotions for signature calculation - Enhanced with tertiary"""
        try:
            with self._lock, sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT emotions FROM messages
                    WHERE session_id = ? AND role = 'user'
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (session_id, limit))
                rows = cursor.fetchall()
                emotions_list = []
                for row in rows:
                    try:
                        emotions = json.loads(row[0]) if row[0] else {}
                        if emotions:
                            # Get top 3 dominant emotions with scores
                            top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                            emotions_list.append({
                                "primary": top_emotions[0][0] if top_emotions else "neutral",
                                "secondary": top_emotions[1][0] if len(top_emotions) > 1 else None,
                                "tertiary": top_emotions[2][0] if len(top_emotions) > 2 else None,
                                "intensities": [v for k, v in top_emotions],
                                "risk_estimate": sum(v for k, v in top_emotions if k in config.emotions_to_monitor)
                            })
                    except json.JSONDecodeError:
                        continue
                return emotions_list
        except sqlite3.Error as e:
            logger.error(f"Error in get_recent_emotions: {e}")
            return []

    def _calculate_emotional_signature(self, recent_emotions: List[Dict]) -> str:
        """Enhanced signature with volatility and trend"""
        if not recent_emotions:
            return "balanced"
        
        # Weighted count of primary emotions
        emotion_counts = {}
        total_intensity = 0
        primaries = [emo.get("primary") for emo in recent_emotions if emo.get("primary")]
        for emo in recent_emotions:
            primary = emo.get("primary")
            intensity = emo.get("intensities", [0])[0]
            if primary:
                emotion_counts[primary] = emotion_counts.get(primary, 0) + intensity
                total_intensity += intensity

        if total_intensity == 0:
            return "neutral"

        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        strength_ratio = emotion_counts[dominant_emotion] / total_intensity

        # Enhanced: Add volatility (number of unique primaries)
        volatility = len(set(primaries))
        trend = "increasing" if len(primaries) > 3 and primaries[-3:] == primaries[:3] else "fluctuating"

        if strength_ratio > 0.6 and volatility < 3:
            return f"persistent_{dominant_emotion}_{trend}"
        elif volatility > 5:
            return "highly_varied"
        else:
            return f"emerging_{dominant_emotion}"

    def get_total_interactions(self, session_id: str) -> int:
        """Get total messages in session for analytics"""
        try:
            with self._lock, sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
                return cursor.fetchone()[0]
        except sqlite3.Error as e:
            logger.error(f"Error counting interactions: {e}")
            return 0

    def update_user_profile(self, session_id: str, preferences: Dict, learned_patterns: List[Dict]):
        """Update user profile with learned preferences and patterns - Enhanced with effectiveness tracking"""
        try:
            profile_data = {
                "preferences": preferences,
                "learned_patterns": learned_patterns[-10:],  # Keep recent 10
                "emotional_signature": self._calculate_emotional_signature(self.get_recent_emotions(session_id)),
                "therapy_style_preference": self._infer_therapy_style(learned_patterns),
                "response_effectiveness": self._calculate_response_effectiveness(learned_patterns),
                "interaction_patterns": self._analyze_interaction_patterns(learned_patterns),
                "last_updated": datetime.now().isoformat()
            }
            
            with self._lock, sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO sessions (session_id, user_profile, emotional_signature, therapy_preference)
                    VALUES (?, ?, ?, ?)
                """, (session_id, json.dumps(profile_data), profile_data["emotional_signature"], preferences.get("therapy_style", "eclectic")))
                conn.commit()
            self.user_profiles[session_id] = profile_data
            logger.info(f"Profile updated for {session_id[:8]}...")
            return profile_data
        except Exception as e:
            logger.error(f"Profile update failed: {e}")
            return None

    def _infer_therapy_style(self, patterns: List[Dict]) -> str:
        """Enhanced inference with more indicators"""
        cbt_indicators = sum(1 for p in patterns if any(word in p.get('input', '').lower() for word in ['think','believe','should','must', 'evidence', 'reframe']))
        mindfulness_indicators = sum(1 for p in patterns if any(word in p.get('input', '').lower() for word in ['feel','notice','present','breathe', 'mindful', 'observe']))
        narrative_indicators = sum(1 for p in patterns if any(word in p.get('input', '').lower() for word in ['story','inspired','memory', 'narrative', 'journey']))
        dbt_indicators = sum(1 for p in patterns if any(word in p.get('input', '').lower() for word in ['accept', 'tolerate', 'distress', 'radical']))
        if cbt_indicators > max(mindfulness_indicators, narrative_indicators, dbt_indicators):
            return "cbt"
        elif mindfulness_indicators > max(narrative_indicators, dbt_indicators):
            return "mindfulness"
        elif dbt_indicators > narrative_indicators:
            return "dbt"
        elif narrative_indicators > 0:
            return "narrative"
        else:
            return "eclectic"

    def _calculate_response_effectiveness(self, patterns: List[Dict]) -> Dict[str, float]:
        """Enhanced with pattern-based scoring"""
        if not patterns:
            return {"empathic_validation": 0.85, "solution_focused": 0.72, "exploratory": 0.78, "inspirational": 0.91}
        
        # Simulate effectiveness from patterns (in real app, use user feedback)
        validations = sum(1 for p in patterns if "validation" in str(p).lower())
        total = len(patterns)
        base_scores = {"empathic_validation": 0.85, "solution_focused": 0.72, "exploratory": 0.78, "inspirational": 0.91}
        base_scores["empathic_validation"] += (validations / total) * 0.15 if total > 0 else 0
        return {k: min(v, 1.0) for k, v in base_scores.items()}

    def _analyze_interaction_patterns(self, patterns: List[Dict]) -> Dict:
        """New: Analyze for recurring themes"""
        themes = defaultdict(int)
        for p in patterns:
            input_text = p.get('input_pattern', '')
            if 'relationship' in input_text.lower():
                themes['relationships'] += 1
            if 'work' in input_text.lower():
                themes['career'] += 1
            if 'self' in input_text.lower():
                themes['self_growth'] += 1
        return dict(themes)

    def new_session(self) -> str:
        """Generate unique session ID with timestamp"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:17]}"

# =====================
# Semantic Memory Class - Enhanced retrieval
# =====================
class SemanticMemory:
    """Stores embeddings of past interactions for contextual retrieval - Enhanced with cosine sim"""
    def __init__(self):
        try:
            # Import lazily here. If sentence-transformers or its heavy
            # dependencies are not available, we gracefully disable semantic memory.
            from sentence_transformers import SentenceTransformer as _ST
            global SentenceTransformer
            SentenceTransformer = _ST
            self.model = _ST('all-MiniLM-L6-v2')
            self.memory_embeddings: Dict[str, List[Dict]] = {}
            self.interaction_count = 0
            logger.info("Semantic memory initialized")
        except Exception as e:
            logger.warning(f"Semantic memory disabled (sentence-transformers unavailable): {e}")
            self.model = None
            self.memory_embeddings = {}
            self.interaction_count = 0

    def store_interaction(self, session_id: str, interaction: str, metadata: Optional[Dict] = None):
        if not self.model:
            return
        try:
            embedding = self.model.encode(interaction)
            if session_id not in self.memory_embeddings:
                self.memory_embeddings[session_id] = []
            self.memory_embeddings[session_id].append({
                "interaction": interaction,
                "embedding": embedding,
                "metadata": metadata or {},
                "timestamp": datetime.now()
            })
            # Keep only last 50 interactions - Enhanced pruning by recency
            self.memory_embeddings[session_id] = sorted(
                self.memory_embeddings[session_id], key=lambda x: x["timestamp"]
            )[-50:]
            self.interaction_count += 1
        except Exception as e:
            logger.error(f"Error storing interaction: {e}")

    def retrieve_relevant_context(self, session_id: str, query: str, top_k: int = 5) -> List[Dict]:
        if not self.model or session_id not in self.memory_embeddings:
            return []
        try:
            query_emb = self.model.encode(query)
            session_memory = self.memory_embeddings[session_id]
            similarities = [(cosine_similarity([query_emb], [item["embedding"]])[0][0], item)
                            for item in session_memory]
            similarities.sort(key=lambda x: x[0], reverse=True)
            relevant = []
            for sim, item in similarities[:top_k]:
                relevant.append({
                    "interaction": item["interaction"],
                    "similarity": float(sim),  # Enhanced with score
                    "metadata": item["metadata"],
                    "timestamp": item["timestamp"]
                })
            return relevant
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []

    def analyze_conversation_patterns(self, session_id: str) -> Dict:
        """Extract conversation patterns and emotional trends - Enhanced volatility calc"""
        if session_id not in self.memory_embeddings:
            return {}
        interactions = self.memory_embeddings[session_id]
        if len(interactions) < 3:
            return {}

        emotional_history = [
            max(i["metadata"].get("emotions", {}), key=i["metadata"]["emotions"].get)
            for i in interactions[-10:] if "emotions" in i["metadata"]
        ]

        # Enhanced volatility: standard deviation of emotion shifts
        if len(emotional_history) >= 5:
            unique_shifts = len(set(emotional_history[-5:]))
            volatility = unique_shifts / 5.0  # Normalized 0-1
        else:
            volatility = 0

        patterns = {
            "emotional_consistency": len(set(emotional_history)) <= 1 if emotional_history else False,
            "recent_dominant_emotion": emotional_history[-1] if emotional_history else None,
            "emotional_volatility": volatility,
            "conversation_depth": len(interactions),
            "preferred_topics": self._extract_topics([i["interaction"] for i in interactions[-5:]]),
            "trend_direction": "stabilizing" if volatility < 0.4 else "fluctuating" if volatility < 0.7 else "volatile"
        }
        return patterns

    def _extract_topics(self, interactions: List[str]) -> List[str]:
        """Enhanced topic extraction with more categories"""
        topic_keywords = {
            'relationships': ['love', 'partner', 'relationship', 'family', 'friend', 'breakup', 'betrayal'],
            'work_career': ['job', 'work', 'career', 'boss', 'colleague', 'stress', 'promotion'],
            'self_growth': ['goal', 'dream', 'purpose', 'confidence', 'motivation', 'growth', 'learning'],
            'health_wellness': ['health', 'exercise', 'diet', 'sleep', 'anxiety', 'depression', 'therapy'],
            'grief_loss': ['loss', 'death', 'grief', 'mourning', 'bereavement'],
            'trauma': ['trauma', 'abuse', 'ptsd', 'flashback', 'trigger']
        }
        detected = []
        for topic, keywords in topic_keywords.items():
            if any(any(kw in msg.lower() for kw in keywords) for msg in interactions):
                detected.append(topic)
        return detected if detected else ["general_reflection"]

# =====================
# Global Instances - Enhanced initialization
# =====================
chat_manager = ChatManager()
semantic_memory = SemanticMemory()

def init_session_profile(session_id: str, initial_preferences: Dict = None):
    """Helper to bootstrap new sessions with profile"""
    if initial_preferences is None:
        initial_preferences = {"therapy_style": "eclectic", "verbosity": 2}
    chat_manager.update_user_profile(session_id, initial_preferences, [])