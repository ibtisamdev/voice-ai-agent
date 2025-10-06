"""
Intent Classification Engine using BERT and rule-based approaches for legal AI agent.
Supports multi-class intent detection with confidence scoring and fallback mechanisms.
"""

import asyncio
import logging
import json
import re
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available - using rule-based classification only")

try:
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available - reduced functionality")

from app.core.config import settings

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    """Main intent categories for legal AI agent."""
    LEGAL_CONSULTATION = "legal_consultation"
    APPOINTMENT_BOOKING = "appointment_booking"
    INFORMATION_REQUEST = "information_request"
    BILLING_INQUIRY = "billing_inquiry"
    CASE_STATUS = "case_status"
    EMERGENCY_LEGAL = "emergency_legal"
    DOCUMENT_REQUEST = "document_request"
    COMPLAINT = "complaint"
    REFERRAL_REQUEST = "referral_request"
    GENERAL_INQUIRY = "general_inquiry"
    SMALL_TALK = "small_talk"
    ESCALATION = "escalation"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    """Intent classification result."""
    name: str
    category: IntentCategory
    confidence: float
    entities: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


@dataclass
class IntentClassificationResult:
    """Complete intent classification result."""
    primary_intent: Intent
    secondary_intents: List[Intent]
    confidence_scores: Dict[str, float]
    processing_time_ms: float
    model_used: str
    timestamp: float
    raw_text: str
    normalized_text: str


class RuleBasedClassifier:
    """Rule-based intent classifier using patterns and keywords."""
    
    def __init__(self):
        self.intent_patterns = self._build_intent_patterns()
        self.entity_patterns = self._build_entity_patterns()
    
    def _build_intent_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build intent patterns with keywords and regular expressions."""
        return {
            IntentCategory.LEGAL_CONSULTATION.value: [
                {
                    'keywords': ['legal', 'lawyer', 'attorney', 'consultation', 'advice', 'help', 'case', 'sue', 'lawsuit'],
                    'patterns': [
                        r'need.*legal.*help',
                        r'speak.*attorney',
                        r'legal.*advice',
                        r'want.*consult',
                        r'have.*legal.*issue',
                        r'need.*lawyer'
                    ],
                    'weight': 1.0
                },
                {
                    'keywords': ['contract', 'agreement', 'review', 'document'],
                    'patterns': [r'review.*contract', r'look.*agreement'],
                    'weight': 0.8
                }
            ],
            IntentCategory.APPOINTMENT_BOOKING.value: [
                {
                    'keywords': ['appointment', 'schedule', 'book', 'meeting', 'available', 'calendar'],
                    'patterns': [
                        r'schedule.*appointment',
                        r'book.*meeting',
                        r'available.*time',
                        r'make.*appointment',
                        r'set.*meeting'
                    ],
                    'weight': 1.0
                }
            ],
            IntentCategory.EMERGENCY_LEGAL.value: [
                {
                    'keywords': ['urgent', 'emergency', 'asap', 'immediately', 'crisis', 'arrest', 'jail'],
                    'patterns': [
                        r'urgent.*matter',
                        r'emergency.*situation',
                        r'need.*immediately',
                        r'been.*arrested',
                        r'in.*jail'
                    ],
                    'weight': 1.0
                }
            ],
            IntentCategory.CASE_STATUS.value: [
                {
                    'keywords': ['status', 'update', 'progress', 'case', 'hearing', 'court'],
                    'patterns': [
                        r'case.*status',
                        r'case.*update',
                        r'court.*date',
                        r'hearing.*scheduled',
                        r'progress.*case'
                    ],
                    'weight': 1.0
                }
            ],
            IntentCategory.BILLING_INQUIRY.value: [
                {
                    'keywords': ['bill', 'invoice', 'payment', 'cost', 'fee', 'charge', 'money'],
                    'patterns': [
                        r'billing.*question',
                        r'payment.*due',
                        r'cost.*services',
                        r'legal.*fees',
                        r'invoice.*received'
                    ],
                    'weight': 1.0
                }
            ],
            IntentCategory.INFORMATION_REQUEST.value: [
                {
                    'keywords': ['information', 'explain', 'understand', 'what', 'how', 'when', 'where'],
                    'patterns': [
                        r'can.*explain',
                        r'how.*does',
                        r'what.*is',
                        r'need.*information',
                        r'tell.*me.*about'
                    ],
                    'weight': 0.7
                }
            ],
            IntentCategory.DOCUMENT_REQUEST.value: [
                {
                    'keywords': ['document', 'paper', 'form', 'copy', 'file', 'send', 'email'],
                    'patterns': [
                        r'send.*document',
                        r'need.*copy',
                        r'email.*file',
                        r'get.*paperwork'
                    ],
                    'weight': 0.9
                }
            ],
            IntentCategory.COMPLAINT.value: [
                {
                    'keywords': ['complain', 'unhappy', 'dissatisfied', 'problem', 'issue', 'wrong'],
                    'patterns': [
                        r'not.*happy',
                        r'complain.*about',
                        r'have.*problem',
                        r'something.*wrong'
                    ],
                    'weight': 0.8
                }
            ],
            IntentCategory.SMALL_TALK.value: [
                {
                    'keywords': ['hello', 'hi', 'good', 'morning', 'afternoon', 'evening', 'thanks', 'thank you'],
                    'patterns': [
                        r'^(hi|hello|hey)',
                        r'good.*(morning|afternoon|evening)',
                        r'thanks?.*you',
                        r'how.*are.*you'
                    ],
                    'weight': 0.6
                }
            ],
            IntentCategory.ESCALATION.value: [
                {
                    'keywords': ['manager', 'supervisor', 'human', 'person', 'transfer', 'escalate'],
                    'patterns': [
                        r'speak.*manager',
                        r'talk.*person',
                        r'transfer.*human',
                        r'escalate.*call'
                    ],
                    'weight': 1.0
                }
            ]
        }
    
    def _build_entity_patterns(self) -> Dict[str, List[str]]:
        """Build entity extraction patterns."""
        return {
            'phone_number': [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',
                r'\+1[-.\s]?\d{3}[-.]?\d{3}[-.]?\d{4}'
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'date': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b'
            ],
            'time': [
                r'\b\d{1,2}:\d{2}\s*(am|pm|AM|PM)?\b',
                r'\b(morning|afternoon|evening|night)\b'
            ],
            'legal_area': [
                r'\b(contract|family|personal injury|criminal|business|real estate|employment|immigration|bankruptcy|divorce|custody|dui|dwi)\s*(law|matter|case|issue)?\b'
            ],
            'urgency': [
                r'\b(urgent|emergency|asap|immediately|right away|as soon as possible)\b'
            ]
        }
    
    def classify(self, text: str) -> IntentClassificationResult:
        """Classify intent using rule-based approach."""
        start_time = time.time()
        normalized_text = self._normalize_text(text)
        
        # Calculate scores for each intent
        intent_scores = {}
        
        for intent_name, patterns_list in self.intent_patterns.items():
            score = 0.0
            
            for pattern_group in patterns_list:
                group_score = 0.0
                
                # Check keywords
                keywords = pattern_group.get('keywords', [])
                keyword_matches = sum(1 for keyword in keywords if keyword.lower() in normalized_text.lower())
                keyword_score = (keyword_matches / len(keywords)) if keywords else 0
                
                # Check regex patterns
                patterns = pattern_group.get('patterns', [])
                pattern_matches = sum(1 for pattern in patterns if re.search(pattern, normalized_text, re.IGNORECASE))
                pattern_score = (pattern_matches / len(patterns)) if patterns else 0
                
                # Combine scores
                group_score = max(keyword_score, pattern_score) * pattern_group.get('weight', 1.0)
                score = max(score, group_score)
            
            intent_scores[intent_name] = score
        
        # Extract entities
        entities = self._extract_entities(text)
        
        # Sort intents by score
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create intent objects
        intents = []
        for intent_name, score in sorted_intents[:3]:  # Top 3 intents
            if score > 0:
                intent = Intent(
                    name=intent_name,
                    category=IntentCategory(intent_name),
                    confidence=min(score, 1.0),
                    entities=entities,
                    description=f"Rule-based classification for {intent_name}"
                )
                intents.append(intent)
        
        # Fallback to unknown if no good match
        if not intents or intents[0].confidence < 0.3:
            unknown_intent = Intent(
                name=IntentCategory.UNKNOWN.value,
                category=IntentCategory.UNKNOWN,
                confidence=0.5,
                entities=entities,
                description="No clear intent detected"
            )
            intents.insert(0, unknown_intent)
        
        primary_intent = intents[0]
        secondary_intents = intents[1:] if len(intents) > 1 else []
        
        processing_time = (time.time() - start_time) * 1000
        
        return IntentClassificationResult(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            confidence_scores=intent_scores,
            processing_time_ms=processing_time,
            model_used="rule_based",
            timestamp=time.time(),
            raw_text=text,
            normalized_text=normalized_text
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove punctuation except periods in numbers
        text = re.sub(r'[^\w\s.]', ' ', text)
        
        return text
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text using regex patterns."""
        entities = {}
        
        for entity_type, patterns in self.entity_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)
            
            if matches:
                entities[entity_type] = list(set(matches))  # Remove duplicates
        
        return entities


class BERTIntentClassifier:
    """BERT-based intent classifier for more sophisticated intent detection."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.embedding_model = None
        self.available = TRANSFORMERS_AVAILABLE
        
        # Intent examples for few-shot learning
        self.intent_examples = self._build_intent_examples()
    
    def _build_intent_examples(self) -> Dict[str, List[str]]:
        """Build training examples for each intent."""
        return {
            IntentCategory.LEGAL_CONSULTATION.value: [
                "I need legal advice about my contract",
                "Can I speak with a lawyer about my case?",
                "I have a legal issue and need help",
                "I want to consult with an attorney",
                "Do I have grounds for a lawsuit?",
                "I need help with a legal matter"
            ],
            IntentCategory.APPOINTMENT_BOOKING.value: [
                "I'd like to schedule an appointment",
                "Can I book a meeting with a lawyer?",
                "What times are available this week?",
                "I need to set up a consultation",
                "When can I meet with an attorney?",
                "Schedule me for next Tuesday"
            ],
            IntentCategory.EMERGENCY_LEGAL.value: [
                "This is urgent, I need help immediately",
                "I have a legal emergency",
                "I was just arrested, need a lawyer now",
                "This can't wait, it's an emergency",
                "I need immediate legal assistance",
                "Crisis situation, need attorney ASAP"
            ],
            IntentCategory.CASE_STATUS.value: [
                "What's the status of my case?",
                "Any updates on my legal matter?",
                "When is my court date?",
                "Has there been progress on my case?",
                "I'm checking on my case status",
                "Any news about my lawsuit?"
            ],
            IntentCategory.BILLING_INQUIRY.value: [
                "I have a question about my bill",
                "How much do I owe for legal services?",
                "Can you explain these charges?",
                "I received an invoice, have questions",
                "What are your legal fees?",
                "Billing question about my account"
            ],
            IntentCategory.INFORMATION_REQUEST.value: [
                "Can you explain how this legal process works?",
                "I need information about divorce procedures",
                "What are my rights in this situation?",
                "Tell me about employment law",
                "How does bankruptcy work?",
                "What should I know about contracts?"
            ],
            IntentCategory.SMALL_TALK.value: [
                "Hello, how are you today?",
                "Good morning",
                "Thank you for your help",
                "Have a great day",
                "Nice to talk with you",
                "Thanks, that's all I needed"
            ],
            IntentCategory.ESCALATION.value: [
                "I want to speak to a manager",
                "Can I talk to a human?",
                "Transfer me to someone else",
                "I need to escalate this call",
                "Let me speak to a supervisor",
                "Connect me with a real person"
            ]
        }
    
    async def initialize(self) -> bool:
        """Initialize BERT models."""
        if not self.available:
            logger.warning("BERT classifier not available - missing transformers")
            return False
        
        try:
            # Load pre-trained model for intent classification
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Load embedding model for similarity
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Pre-compute embeddings for intent examples
            self._precompute_embeddings()
            
            logger.info("BERT intent classifier initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize BERT classifier: {e}")
            self.available = False
            return False
    
    def _precompute_embeddings(self):
        """Pre-compute embeddings for intent examples."""
        self.intent_embeddings = {}
        
        for intent, examples in self.intent_examples.items():
            embeddings = self.embedding_model.encode(examples)
            self.intent_embeddings[intent] = embeddings
    
    async def classify(self, text: str) -> IntentClassificationResult:
        """Classify intent using BERT models."""
        if not self.available:
            raise Exception("BERT classifier not available")
        
        start_time = time.time()
        
        # Get candidate labels
        candidate_labels = list(self.intent_examples.keys())
        
        # Zero-shot classification
        result = self.classifier(text, candidate_labels)
        
        # Create intents from results
        intents = []
        confidence_scores = {}
        
        for label, score in zip(result['labels'], result['scores']):
            intent = Intent(
                name=label,
                category=IntentCategory(label),
                confidence=float(score),
                description=f"BERT classification for {label}"
            )
            intents.append(intent)
            confidence_scores[label] = float(score)
        
        # Enhance with similarity scoring
        if self.embedding_model:
            similarity_scores = await self._calculate_similarity_scores(text)
            
            # Combine BERT and similarity scores
            for intent in intents:
                bert_score = intent.confidence
                similarity_score = similarity_scores.get(intent.name, 0.0)
                # Weighted combination
                combined_score = 0.7 * bert_score + 0.3 * similarity_score
                intent.confidence = combined_score
                confidence_scores[intent.name] = combined_score
        
        # Re-sort by combined confidence
        intents.sort(key=lambda x: x.confidence, reverse=True)
        
        primary_intent = intents[0] if intents else Intent(
            name=IntentCategory.UNKNOWN.value,
            category=IntentCategory.UNKNOWN,
            confidence=0.5
        )
        
        secondary_intents = intents[1:3] if len(intents) > 1 else []
        
        processing_time = (time.time() - start_time) * 1000
        
        return IntentClassificationResult(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            confidence_scores=confidence_scores,
            processing_time_ms=processing_time,
            model_used="bert",
            timestamp=time.time(),
            raw_text=text,
            normalized_text=text.lower().strip()
        )
    
    async def _calculate_similarity_scores(self, text: str) -> Dict[str, float]:
        """Calculate similarity scores using sentence embeddings."""
        try:
            # Get embedding for input text
            text_embedding = self.embedding_model.encode([text])
            
            similarity_scores = {}
            
            for intent, intent_embeddings in self.intent_embeddings.items():
                # Calculate cosine similarities
                similarities = np.dot(text_embedding, intent_embeddings.T) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(intent_embeddings, axis=1)
                )
                
                # Use maximum similarity as the score
                max_similarity = float(np.max(similarities))
                similarity_scores[intent] = max(0.0, max_similarity)
            
            return similarity_scores
            
        except Exception as e:
            logger.error(f"Error calculating similarity scores: {e}")
            return {}


class IntentClassifier:
    """Main intent classifier combining multiple approaches."""
    
    def __init__(self):
        self.rule_based_classifier = RuleBasedClassifier()
        self.bert_classifier = BERTIntentClassifier()
        self.use_bert = False
        
        # Statistics
        self.stats = {
            "classifications_completed": 0,
            "average_confidence": 0.0,
            "model_usage": {"rule_based": 0, "bert": 0, "hybrid": 0},
            "intent_distribution": {},
            "processing_time_ms": 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize the intent classifier."""
        try:
            logger.info("Initializing intent classifier...")
            
            # Try to initialize BERT classifier
            if await self.bert_classifier.initialize():
                self.use_bert = True
                logger.info("BERT classifier available")
            else:
                logger.info("Using rule-based classifier only")
            
            logger.info("Intent classifier initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize intent classifier: {e}")
            return False
    
    async def classify_intent(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        use_bert: Optional[bool] = None
    ) -> IntentClassificationResult:
        """
        Classify intent from text.
        
        Args:
            text: Input text to classify
            context: Optional conversation context
            use_bert: Force BERT usage (if available)
        
        Returns:
            IntentClassificationResult: Classification result
        """
        try:
            # Determine which classifier to use
            should_use_bert = (use_bert if use_bert is not None else self.use_bert)
            
            if should_use_bert and self.bert_classifier.available:
                # Use BERT classifier
                result = await self.bert_classifier.classify(text)
                model_used = "bert"
            else:
                # Use rule-based classifier
                result = self.rule_based_classifier.classify(text)
                model_used = "rule_based"
            
            # Enhance with context if provided
            if context:
                result = self._enhance_with_context(result, context)
            
            # Update statistics
            self._update_stats(result, model_used)
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            # Return fallback result
            return IntentClassificationResult(
                primary_intent=Intent(
                    name=IntentCategory.UNKNOWN.value,
                    category=IntentCategory.UNKNOWN,
                    confidence=0.5,
                    description="Error in classification"
                ),
                secondary_intents=[],
                confidence_scores={IntentCategory.UNKNOWN.value: 0.5},
                processing_time_ms=0.0,
                model_used="error_fallback",
                timestamp=time.time(),
                raw_text=text,
                normalized_text=text.lower()
            )
    
    def _enhance_with_context(
        self, 
        result: IntentClassificationResult, 
        context: Dict[str, Any]
    ) -> IntentClassificationResult:
        """Enhance classification result with conversation context."""
        try:
            # Adjust confidence based on context
            if 'previous_intent' in context:
                prev_intent = context['previous_intent']
                
                # Boost confidence if same intent as previous
                if result.primary_intent.name == prev_intent:
                    result.primary_intent.confidence = min(1.0, result.primary_intent.confidence * 1.2)
            
            # Check for escalation indicators in context
            if context.get('user_frustrated', False):
                # Boost escalation intent if user seems frustrated
                for intent in [result.primary_intent] + result.secondary_intents:
                    if intent.category == IntentCategory.ESCALATION:
                        intent.confidence = min(1.0, intent.confidence * 1.5)
                        break
            
            # Add context to primary intent
            if result.primary_intent.context is None:
                result.primary_intent.context = {}
            result.primary_intent.context.update(context)
            
            return result
            
        except Exception as e:
            logger.error(f"Error enhancing with context: {e}")
            return result
    
    def _update_stats(self, result: IntentClassificationResult, model_used: str):
        """Update classification statistics."""
        try:
            self.stats["classifications_completed"] += 1
            self.stats["model_usage"][model_used] += 1
            
            # Update average confidence
            total_confidence = self.stats["average_confidence"] * (self.stats["classifications_completed"] - 1)
            self.stats["average_confidence"] = (total_confidence + result.primary_intent.confidence) / self.stats["classifications_completed"]
            
            # Update intent distribution
            intent_name = result.primary_intent.name
            self.stats["intent_distribution"][intent_name] = self.stats["intent_distribution"].get(intent_name, 0) + 1
            
            # Update processing time
            total_time = self.stats["processing_time_ms"] * (self.stats["classifications_completed"] - 1)
            self.stats["processing_time_ms"] = (total_time + result.processing_time_ms) / self.stats["classifications_completed"]
            
        except Exception as e:
            logger.error(f"Error updating stats: {e}")
    
    def get_available_intents(self) -> List[str]:
        """Get list of available intent categories."""
        return [intent.value for intent in IntentCategory]
    
    def get_intent_description(self, intent_name: str) -> Optional[str]:
        """Get description for an intent."""
        descriptions = {
            IntentCategory.LEGAL_CONSULTATION.value: "User needs legal advice or consultation",
            IntentCategory.APPOINTMENT_BOOKING.value: "User wants to schedule an appointment",
            IntentCategory.INFORMATION_REQUEST.value: "User is requesting information",
            IntentCategory.BILLING_INQUIRY.value: "User has billing or payment questions",
            IntentCategory.CASE_STATUS.value: "User wants case status update",
            IntentCategory.EMERGENCY_LEGAL.value: "Urgent legal matter requiring immediate attention",
            IntentCategory.DOCUMENT_REQUEST.value: "User needs documents or forms",
            IntentCategory.COMPLAINT.value: "User has a complaint or issue",
            IntentCategory.REFERRAL_REQUEST.value: "User needs referral to another service",
            IntentCategory.GENERAL_INQUIRY.value: "General inquiry about services",
            IntentCategory.SMALL_TALK.value: "Social conversation, greetings",
            IntentCategory.ESCALATION.value: "User wants to speak to human agent",
            IntentCategory.UNKNOWN.value: "Intent could not be determined"
        }
        return descriptions.get(intent_name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get classification statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset classification statistics."""
        self.stats = {
            "classifications_completed": 0,
            "average_confidence": 0.0,
            "model_usage": {"rule_based": 0, "bert": 0, "hybrid": 0},
            "intent_distribution": {},
            "processing_time_ms": 0.0
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of intent classifier."""
        try:
            # Test classification
            test_result = await self.classify_intent("Hello, I need legal help")
            
            return {
                "healthy": True,
                "rule_based_available": True,
                "bert_available": self.bert_classifier.available,
                "current_model": "bert" if self.use_bert else "rule_based",
                "available_intents": len(self.get_available_intents()),
                "test_classification": {
                    "intent": test_result.primary_intent.name,
                    "confidence": test_result.primary_intent.confidence,
                    "processing_time_ms": test_result.processing_time_ms
                },
                "stats": self.get_stats(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Intent classifier health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global intent classifier instance
intent_classifier = IntentClassifier()