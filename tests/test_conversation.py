"""Tests for conversation management and dialog flow."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
import json
from datetime import datetime, timedelta

from ai.conversation.state_manager import ConversationStateManager, ConversationSession
from ai.conversation.dialog_flow import DialogFlowEngine, FlowNode
from ai.decision_engine.intent_classifier import IntentClassifier


class TestConversationSession:
    """Test conversation session data model."""
    
    def test_session_creation(self):
        """Test creating a new conversation session."""
        session = ConversationSession(
            session_id="test_session_123",
            session_type="voice_call",
            direction="inbound"
        )
        
        assert session.session_id == "test_session_123"
        assert session.session_type == "voice_call"
        assert session.direction == "inbound"
        assert session.state == "active"
        assert len(session.turns) == 0
        assert session.created_at is not None
    
    def test_add_turn(self):
        """Test adding turns to a session."""
        session = ConversationSession(
            session_id="test_session_456",
            session_type="voice_call",
            direction="inbound"
        )
        
        # Add user turn
        session.add_turn(
            speaker_id="user",
            speaker_role="customer",
            input_text="Hello, I need legal help",
            intent="legal_consultation",
            intent_confidence=0.95
        )
        
        assert len(session.turns) == 1
        turn = session.turns[0]
        assert turn['speaker_id'] == "user"
        assert turn['speaker_role'] == "customer"
        assert turn['input_text'] == "Hello, I need legal help"
        assert turn['intent'] == "legal_consultation"
        assert turn['intent_confidence'] == 0.95
        assert turn['turn_index'] == 0
    
    def test_session_context(self):
        """Test session context management."""
        session = ConversationSession(
            session_id="test_context",
            session_type="voice_call",
            direction="inbound"
        )
        
        # Set context
        session.set_context("case_type", "personal_injury")
        session.set_context("urgency", "high")
        
        assert session.get_context("case_type") == "personal_injury"
        assert session.get_context("urgency") == "high"
        assert session.get_context("nonexistent") is None
    
    def test_session_serialization(self):
        """Test session serialization for storage."""
        session = ConversationSession(
            session_id="test_serialize",
            session_type="voice_call",
            direction="inbound"
        )
        
        session.add_turn(
            speaker_id="user",
            speaker_role="customer",
            input_text="Test message",
            intent="general_inquiry"
        )
        
        session.set_context("test_key", "test_value")
        
        # Serialize to dict
        session_dict = session.to_dict()
        
        assert session_dict['session_id'] == "test_serialize"
        assert len(session_dict['turns']) == 1
        assert session_dict['context']['test_key'] == "test_value"
        
        # Deserialize from dict
        restored_session = ConversationSession.from_dict(session_dict)
        
        assert restored_session.session_id == session.session_id
        assert len(restored_session.turns) == 1
        assert restored_session.get_context("test_key") == "test_value"


class TestConversationStateManager:
    """Test conversation state management."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis_client = Mock()
        redis_client.get.return_value = None
        redis_client.set.return_value = True
        redis_client.delete.return_value = True
        redis_client.exists.return_value = True
        redis_client.expire.return_value = True
        return redis_client
    
    @pytest.fixture
    def state_manager(self, mock_redis):
        """Create state manager with mocked Redis."""
        return ConversationStateManager(redis_client=mock_redis)
    
    @pytest.mark.asyncio
    async def test_create_session(self, state_manager):
        """Test creating a new conversation session."""
        session_id = "test_session_create"
        
        session = await state_manager.create_session(
            session_id=session_id,
            session_type="voice_call",
            direction="inbound",
            phone_number="+1234567890"
        )
        
        assert session.session_id == session_id
        assert session.session_type == "voice_call"
        assert session.direction == "inbound"
        assert session.phone_number == "+1234567890"
        
        # Verify Redis was called to store session
        state_manager.redis_client.set.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_session(self, state_manager, mock_redis):
        """Test retrieving an existing session."""
        session_id = "test_session_get"
        
        # Mock existing session data
        session_data = {
            'session_id': session_id,
            'session_type': 'voice_call',
            'direction': 'inbound',
            'state': 'active',
            'turns': [],
            'context': {},
            'created_at': datetime.now().isoformat()
        }
        
        mock_redis.get.return_value = json.dumps(session_data)
        
        session = await state_manager.get_session(session_id)
        
        assert session is not None
        assert session.session_id == session_id
        assert session.session_type == "voice_call"
        
        mock_redis.get.assert_called_with(f"conversation:session:{session_id}")
    
    @pytest.mark.asyncio
    async def test_update_session(self, state_manager):
        """Test updating an existing session."""
        session_id = "test_session_update"
        
        # Create session
        session = await state_manager.create_session(
            session_id=session_id,
            session_type="voice_call",
            direction="inbound"
        )
        
        # Add turn
        session.add_turn(
            speaker_id="user",
            speaker_role="customer",
            input_text="I need help with a contract",
            intent="contract_review"
        )
        
        # Update session
        await state_manager.update_session(session)
        
        # Verify Redis set was called multiple times (create + update)
        assert state_manager.redis_client.set.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_end_session(self, state_manager):
        """Test ending a conversation session."""
        session_id = "test_session_end"
        
        # Create session
        session = await state_manager.create_session(
            session_id=session_id,
            session_type="voice_call",
            direction="inbound"
        )
        
        # End session
        await state_manager.end_session(session_id)
        
        # Session should be marked as ended
        updated_session = await state_manager.get_session(session_id)
        # In real implementation, this would update the state
        
        state_manager.redis_client.set.assert_called()
    
    @pytest.mark.asyncio
    async def test_session_cleanup(self, state_manager):
        """Test automatic session cleanup."""
        expired_sessions = [
            "expired_session_1",
            "expired_session_2"
        ]
        
        # Mock finding expired sessions
        with patch.object(state_manager, 'find_expired_sessions', return_value=expired_sessions):
            cleaned_count = await state_manager.cleanup_expired_sessions()
            
            assert cleaned_count == 2
            
            # Verify cleanup was called for each expired session
            assert state_manager.redis_client.delete.call_count == 2
    
    @pytest.mark.asyncio
    async def test_session_statistics(self, state_manager):
        """Test getting session statistics."""
        with patch.object(state_manager, 'get_active_session_count', return_value=5), \
             patch.object(state_manager, 'get_total_sessions_today', return_value=25):
            
            stats = await state_manager.get_session_statistics()
            
            assert stats['active_sessions'] == 5
            assert stats['total_sessions_today'] == 25
            assert 'timestamp' in stats


class TestDialogFlowEngine:
    """Test dialog flow management."""
    
    @pytest.fixture
    def sample_flow_config(self):
        """Sample dialog flow configuration."""
        return {
            'name': 'Legal Consultation Flow',
            'version': '1.0',
            'nodes': [
                {
                    'id': 'greeting',
                    'type': 'response',
                    'content': 'Hello! How can I help you today?',
                    'next': 'intent_classification'
                },
                {
                    'id': 'intent_classification',
                    'type': 'intent',
                    'intents': ['legal_consultation', 'appointment_booking'],
                    'default': 'general_inquiry'
                },
                {
                    'id': 'legal_consultation',
                    'type': 'response',
                    'content': 'I understand you need legal consultation.',
                    'next': 'collect_case_details'
                },
                {
                    'id': 'collect_case_details',
                    'type': 'slot_filling',
                    'slots': [
                        {'name': 'case_type', 'prompt': 'What type of case is this?', 'required': True},
                        {'name': 'urgency', 'prompt': 'How urgent is this matter?', 'required': False}
                    ],
                    'next': 'end'
                },
                {
                    'id': 'end',
                    'type': 'end',
                    'content': 'Thank you for contacting us.'
                }
            ]
        }
    
    @pytest.fixture
    def dialog_engine(self, sample_flow_config):
        """Create dialog flow engine with sample configuration."""
        return DialogFlowEngine(flow_config=sample_flow_config)
    
    def test_engine_initialization(self, dialog_engine, sample_flow_config):
        """Test dialog flow engine initialization."""
        assert dialog_engine.flow_name == 'Legal Consultation Flow'
        assert len(dialog_engine.nodes) == len(sample_flow_config['nodes'])
        assert 'greeting' in dialog_engine.nodes
        assert 'intent_classification' in dialog_engine.nodes
    
    def test_start_flow(self, dialog_engine):
        """Test starting a dialog flow."""
        session_id = "test_flow_session"
        
        result = dialog_engine.start_flow(session_id)
        
        assert result['session_id'] == session_id
        assert result['current_node'] == 'greeting'
        assert result['response'] == 'Hello! How can I help you today?'
        assert result['next_action'] == 'wait_for_input'
    
    def test_process_user_input(self, dialog_engine):
        """Test processing user input in dialog flow."""
        session_id = "test_input_session"
        
        # Start flow
        dialog_engine.start_flow(session_id)
        
        # Process user input
        user_input = "I need legal consultation"
        
        with patch.object(dialog_engine, 'classify_intent', return_value='legal_consultation'):
            result = dialog_engine.process_input(session_id, user_input)
            
            assert result['current_node'] == 'legal_consultation'
            assert 'I understand you need legal consultation' in result['response']
    
    def test_slot_filling(self, dialog_engine):
        """Test slot filling functionality."""
        session_id = "test_slot_session"
        
        # Navigate to slot filling node
        dialog_engine.start_flow(session_id)
        dialog_engine.set_current_node(session_id, 'collect_case_details')
        
        # Fill first slot
        result1 = dialog_engine.process_input(session_id, "personal injury")
        assert 'case_type' in dialog_engine.get_session_context(session_id)
        
        # Fill second slot
        result2 = dialog_engine.process_input(session_id, "high urgency")
        assert 'urgency' in dialog_engine.get_session_context(session_id)
    
    def test_conditional_branching(self, dialog_engine):
        """Test conditional branching in dialog flow."""
        session_id = "test_branch_session"
        
        # Add a conditional node to the flow
        conditional_node = {
            'id': 'conditional_test',
            'type': 'conditional',
            'condition': 'context.case_type == "personal_injury"',
            'true_next': 'personal_injury_flow',
            'false_next': 'general_flow'
        }
        
        dialog_engine.nodes['conditional_test'] = FlowNode(conditional_node)
        
        # Set context and test branching
        dialog_engine.start_flow(session_id)
        dialog_engine.set_context(session_id, 'case_type', 'personal_injury')
        dialog_engine.set_current_node(session_id, 'conditional_test')
        
        result = dialog_engine.process_node(session_id)
        
        # Should branch to personal_injury_flow
        assert result['next_node'] == 'personal_injury_flow'
    
    def test_flow_validation(self, dialog_engine):
        """Test dialog flow validation."""
        # Test valid flow
        assert dialog_engine.validate_flow()
        
        # Test invalid flow (missing node reference)
        invalid_config = {
            'name': 'Invalid Flow',
            'nodes': [
                {
                    'id': 'start',
                    'type': 'response',
                    'content': 'Hello',
                    'next': 'nonexistent_node'  # Invalid reference
                }
            ]
        }
        
        invalid_engine = DialogFlowEngine(flow_config=invalid_config)
        assert not invalid_engine.validate_flow()
    
    def test_flow_state_persistence(self, dialog_engine):
        """Test dialog flow state persistence."""
        session_id = "test_persistence"
        
        # Start flow and make some progress
        dialog_engine.start_flow(session_id)
        dialog_engine.set_context(session_id, 'test_key', 'test_value')
        
        # Get flow state
        state = dialog_engine.get_flow_state(session_id)
        
        assert state['current_node'] == 'greeting'
        assert state['context']['test_key'] == 'test_value'
        
        # Restore flow state
        new_engine = DialogFlowEngine(flow_config=dialog_engine.flow_config)
        new_engine.restore_flow_state(session_id, state)
        
        assert new_engine.get_current_node(session_id) == 'greeting'
        assert new_engine.get_context(session_id, 'test_key') == 'test_value'


class TestIntentClassifier:
    """Test intent classification."""
    
    @pytest.fixture
    def mock_bert_model(self):
        """Mock BERT model for intent classification."""
        model = Mock()
        tokenizer = Mock()
        
        # Mock tokenizer output
        tokenizer.return_value = {
            'input_ids': [[101, 7592, 2003, 1037, 3231, 102]],
            'attention_mask': [[1, 1, 1, 1, 1, 1]]
        }
        
        # Mock model output
        model.return_value = Mock(logits=[[0.1, 0.8, 0.05, 0.05]])  # High confidence for intent 1
        
        return model, tokenizer
    
    @pytest.fixture
    def intent_classifier(self, mock_bert_model):
        """Create intent classifier with mocked BERT model."""
        model, tokenizer = mock_bert_model
        
        with patch('ai.decision_engine.intent_classifier.AutoModel.from_pretrained', return_value=model), \
             patch('ai.decision_engine.intent_classifier.AutoTokenizer.from_pretrained', return_value=tokenizer):
            
            classifier = IntentClassifier()
            return classifier
    
    def test_classifier_initialization(self, intent_classifier):
        """Test intent classifier initialization."""
        assert intent_classifier.model is not None
        assert intent_classifier.tokenizer is not None
        assert len(intent_classifier.intent_labels) > 0
    
    def test_text_classification(self, intent_classifier):
        """Test text intent classification."""
        text = "I need help with a legal consultation"
        
        result = intent_classifier.classify(text)
        
        assert 'intent' in result
        assert 'confidence' in result
        assert isinstance(result['confidence'], float)
        assert 0 <= result['confidence'] <= 1
    
    def test_rule_based_classification(self, intent_classifier):
        """Test rule-based intent classification."""
        # Test appointment booking keywords
        appointment_text = "I want to schedule an appointment for next week"
        result = intent_classifier.classify_rule_based(appointment_text)
        assert result['intent'] == 'appointment_booking'
        
        # Test legal consultation keywords
        legal_text = "I need legal advice about my contract"
        result = intent_classifier.classify_rule_based(legal_text)
        assert result['intent'] == 'legal_consultation'
        
        # Test general inquiry fallback
        general_text = "Hello there"
        result = intent_classifier.classify_rule_based(general_text)
        assert result['intent'] == 'general_inquiry'
    
    def test_hybrid_classification(self, intent_classifier):
        """Test hybrid classification (BERT + rules)."""
        text = "I need to book an appointment for legal consultation"
        
        result = intent_classifier.classify_hybrid(text)
        
        assert 'intent' in result
        assert 'confidence' in result
        assert 'method' in result  # Should indicate which method was used
    
    def test_batch_classification(self, intent_classifier):
        """Test batch intent classification."""
        texts = [
            "I need legal help",
            "Schedule an appointment",
            "What are your hours?"
        ]
        
        results = intent_classifier.classify_batch(texts)
        
        assert len(results) == 3
        for result in results:
            assert 'intent' in result
            assert 'confidence' in result
    
    def test_confidence_thresholding(self, intent_classifier):
        """Test confidence thresholding for intent classification."""
        # Set high confidence threshold
        intent_classifier.confidence_threshold = 0.9
        
        # Mock low confidence result
        with patch.object(intent_classifier, 'classify_bert') as mock_classify:
            mock_classify.return_value = {'intent': 'legal_consultation', 'confidence': 0.5}
            
            text = "Ambiguous text"
            result = intent_classifier.classify(text)
            
            # Should fall back to rule-based or return low confidence
            assert result['confidence'] < 0.9 or result['intent'] == 'general_inquiry'
    
    def test_entity_extraction(self, intent_classifier):
        """Test entity extraction from text."""
        text = "I need an appointment on Monday at 3 PM for contract review"
        
        entities = intent_classifier.extract_entities(text)
        
        assert isinstance(entities, dict)
        # Should extract date, time, and service type entities
        # Note: This would require NER model implementation


@pytest.mark.integration
class TestConversationIntegration:
    """Integration tests for conversation components."""
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test complete conversation flow end-to-end."""
        # Mock dependencies
        mock_redis = Mock()
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        
        # Create components
        state_manager = ConversationStateManager(redis_client=mock_redis)
        
        sample_flow = {
            'name': 'Test Flow',
            'nodes': [
                {
                    'id': 'greeting',
                    'type': 'response',
                    'content': 'Hello! How can I help?',
                    'next': 'intent_classification'
                },
                {
                    'id': 'intent_classification',
                    'type': 'intent',
                    'intents': ['legal_consultation'],
                    'default': 'general_inquiry'
                },
                {
                    'id': 'legal_consultation',
                    'type': 'response',
                    'content': 'I can help with legal matters.',
                    'next': 'end'
                },
                {
                    'id': 'end',
                    'type': 'end',
                    'content': 'Thank you!'
                }
            ]
        }
        
        dialog_engine = DialogFlowEngine(flow_config=sample_flow)
        
        # Create session
        session = await state_manager.create_session(
            session_id="integration_test",
            session_type="voice_call",
            direction="inbound"
        )
        
        # Start dialog flow
        flow_result = dialog_engine.start_flow(session.session_id)
        
        # Add turn to session
        session.add_turn(
            speaker_id="ai",
            speaker_role="assistant",
            response_text=flow_result['response']
        )
        
        # Process user input
        with patch.object(dialog_engine, 'classify_intent', return_value='legal_consultation'):
            user_input = "I need legal help"
            
            session.add_turn(
                speaker_id="user",
                speaker_role="customer",
                input_text=user_input,
                intent="legal_consultation"
            )
            
            flow_result = dialog_engine.process_input(session.session_id, user_input)
            
            session.add_turn(
                speaker_id="ai",
                speaker_role="assistant",
                response_text=flow_result['response']
            )
        
        # Update session
        await state_manager.update_session(session)
        
        # Verify conversation state
        assert len(session.turns) == 3  # AI greeting, user input, AI response
        assert session.turns[0]['speaker_role'] == 'assistant'
        assert session.turns[1]['speaker_role'] == 'customer'
        assert session.turns[1]['intent'] == 'legal_consultation'
        assert session.turns[2]['speaker_role'] == 'assistant'
    
    @pytest.mark.asyncio
    async def test_conversation_error_handling(self):
        """Test error handling in conversation flow."""
        mock_redis = Mock()
        mock_redis.get.side_effect = Exception("Redis connection error")
        
        state_manager = ConversationStateManager(redis_client=mock_redis)
        
        # Should handle Redis errors gracefully
        with pytest.raises(Exception):
            await state_manager.get_session("test_session")
    
    @pytest.mark.asyncio
    async def test_conversation_performance(self):
        """Test conversation processing performance."""
        import time
        
        mock_redis = Mock()
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        
        state_manager = ConversationStateManager(redis_client=mock_redis)
        
        # Measure session creation time
        start_time = time.time()
        
        tasks = []
        for i in range(10):
            task = state_manager.create_session(
                session_id=f"perf_test_{i}",
                session_type="voice_call",
                direction="inbound"
            )
            tasks.append(task)
        
        sessions = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # Should create 10 sessions quickly
        assert len(sessions) == 10
        assert (end_time - start_time) < 1.0  # Should complete in under 1 second