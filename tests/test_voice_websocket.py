"""Tests for WebSocket voice streaming endpoints."""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect
import numpy as np

from backend.app.api.v1.voice_ws import VoiceSessionHandler
from backend.app.main import app


class TestVoiceSessionHandler:
    """Test WebSocket voice session handler."""
    
    @pytest.fixture
    def mock_services(self):
        """Mock voice processing services."""
        services = {
            'stt_service': Mock(),
            'tts_service': Mock(),
            'conversation_state': Mock(),
            'dialog_engine': Mock(),
            'intent_classifier': Mock()
        }
        
        # Configure mock behaviors
        services['stt_service'].create_session.return_value = {
            'session_id': 'test_session',
            'status': 'active'
        }
        services['stt_service'].process_audio_chunk.return_value = {
            'partial_text': 'Hello',
            'is_final': False
        }
        services['stt_service'].get_final_result.return_value = {
            'text': 'Hello, I need legal help',
            'confidence': 0.95,
            'language': 'en'
        }
        
        services['tts_service'].synthesize_async = AsyncMock(return_value={
            'audio': b'fake_audio_data',
            'metadata': {'duration': 2.5, 'format': 'wav'}
        })
        
        services['conversation_state'].create_session = AsyncMock(return_value=Mock(
            session_id='test_session',
            add_turn=Mock(),
            to_dict=Mock(return_value={'session_id': 'test_session'})
        ))
        
        services['dialog_engine'].process_input.return_value = {
            'response': 'How can I help you with your legal matter?',
            'next_action': 'wait_for_input',
            'current_node': 'legal_consultation'
        }
        
        services['intent_classifier'].classify.return_value = {
            'intent': 'legal_consultation',
            'confidence': 0.9
        }
        
        return services
    
    @pytest.fixture
    def session_handler(self, mock_services):
        """Create voice session handler with mocked services."""
        handler = VoiceSessionHandler(
            session_id="test_session_123",
            stt_service=mock_services['stt_service'],
            tts_service=mock_services['tts_service'],
            conversation_state=mock_services['conversation_state'],
            dialog_engine=mock_services['dialog_engine'],
            intent_classifier=mock_services['intent_classifier']
        )
        return handler
    
    @pytest.mark.asyncio
    async def test_session_initialization(self, session_handler):
        """Test WebSocket session initialization."""
        mock_websocket = Mock()
        
        await session_handler.initialize_session(mock_websocket)
        
        assert session_handler.websocket == mock_websocket
        assert session_handler.is_active
        session_handler.stt_service.create_session.assert_called_once()
        session_handler.conversation_state.create_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_audio_message_processing(self, session_handler):
        """Test processing audio messages."""
        mock_websocket = Mock()
        session_handler.websocket = mock_websocket
        session_handler.is_active = True
        
        # Mock audio data
        audio_data = np.random.random(1600).astype(np.float32)  # 0.1 seconds
        audio_bytes = audio_data.tobytes()
        
        message = {
            'type': 'audio',
            'data': audio_bytes.hex(),  # Hex encoded audio
            'format': 'raw_float32',
            'sample_rate': 16000
        }
        
        await session_handler.handle_message(json.dumps(message))
        
        # Should process audio chunk
        session_handler.stt_service.process_audio_chunk.assert_called()
    
    @pytest.mark.asyncio
    async def test_text_message_processing(self, session_handler):
        """Test processing text messages."""
        mock_websocket = Mock()
        mock_websocket.send_text = AsyncMock()
        session_handler.websocket = mock_websocket
        session_handler.is_active = True
        
        message = {
            'type': 'text',
            'text': 'Hello, I need legal consultation'
        }
        
        await session_handler.handle_message(json.dumps(message))
        
        # Should classify intent and process through dialog engine
        session_handler.intent_classifier.classify.assert_called_with('Hello, I need legal consultation')
        session_handler.dialog_engine.process_input.assert_called()
        
        # Should send response back
        mock_websocket.send_text.assert_called()
    
    @pytest.mark.asyncio
    async def test_voice_activity_detection(self, session_handler):
        """Test voice activity detection in audio stream."""
        mock_websocket = Mock()
        mock_websocket.send_text = AsyncMock()
        session_handler.websocket = mock_websocket
        session_handler.is_active = True
        
        # Mock silence detection
        session_handler.stt_service.process_audio_chunk.return_value = {
            'partial_text': '',
            'is_final': False,
            'voice_detected': False
        }
        
        silent_audio = np.zeros(1600, dtype=np.float32)
        audio_bytes = silent_audio.tobytes()
        
        message = {
            'type': 'audio',
            'data': audio_bytes.hex(),
            'format': 'raw_float32',
            'sample_rate': 16000
        }
        
        await session_handler.handle_message(json.dumps(message))
        
        # Should not trigger speech processing for silence
        session_handler.dialog_engine.process_input.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_final_transcription_processing(self, session_handler):
        """Test processing final transcription results."""
        mock_websocket = Mock()
        mock_websocket.send_text = AsyncMock()
        mock_websocket.send_bytes = AsyncMock()
        session_handler.websocket = mock_websocket
        session_handler.is_active = True
        
        # Mock final transcription result
        session_handler.stt_service.get_final_result.return_value = {
            'text': 'I need help with a contract review',
            'confidence': 0.95,
            'language': 'en',
            'is_final': True
        }
        
        await session_handler.process_final_transcription()
        
        # Should process through conversation flow
        session_handler.intent_classifier.classify.assert_called()
        session_handler.dialog_engine.process_input.assert_called()
        session_handler.tts_service.synthesize_async.assert_called()
        
        # Should send text and audio responses
        mock_websocket.send_text.assert_called()
        mock_websocket.send_bytes.assert_called()
    
    @pytest.mark.asyncio
    async def test_connection_management(self, session_handler):
        """Test WebSocket connection management."""
        mock_websocket = Mock()
        session_handler.websocket = mock_websocket
        
        # Test connection
        await session_handler.initialize_session(mock_websocket)
        assert session_handler.is_active
        
        # Test disconnection
        await session_handler.handle_disconnect()
        assert not session_handler.is_active
        
        # Should cleanup resources
        session_handler.stt_service.end_session.assert_called()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, session_handler):
        """Test error handling in WebSocket session."""
        mock_websocket = Mock()
        mock_websocket.send_text = AsyncMock()
        session_handler.websocket = mock_websocket
        session_handler.is_active = True
        
        # Mock service error
        session_handler.stt_service.process_audio_chunk.side_effect = Exception("STT Error")
        
        audio_data = np.random.random(1600).astype(np.float32)
        message = {
            'type': 'audio',
            'data': audio_data.tobytes().hex(),
            'format': 'raw_float32',
            'sample_rate': 16000
        }
        
        await session_handler.handle_message(json.dumps(message))
        
        # Should send error message to client
        mock_websocket.send_text.assert_called()
        call_args = mock_websocket.send_text.call_args[0][0]
        error_message = json.loads(call_args)
        assert error_message['type'] == 'error'
    
    @pytest.mark.asyncio
    async def test_session_state_persistence(self, session_handler):
        """Test session state persistence during conversation."""
        mock_websocket = Mock()
        mock_websocket.send_text = AsyncMock()
        session_handler.websocket = mock_websocket
        session_handler.is_active = True
        
        # Process multiple turns
        messages = [
            "Hello, I need legal help",
            "It's about a contract dispute",
            "The contract was signed last month"
        ]
        
        for i, text in enumerate(messages):
            message = {'type': 'text', 'text': text}
            await session_handler.handle_message(json.dumps(message))
            
            # Each turn should be added to session
            assert session_handler.conversation_state.create_session.return_value.add_turn.call_count >= i + 1
    
    @pytest.mark.asyncio
    async def test_audio_format_handling(self, session_handler):
        """Test handling different audio formats."""
        mock_websocket = Mock()
        session_handler.websocket = mock_websocket
        session_handler.is_active = True
        
        # Test different audio formats
        audio_formats = [
            {'format': 'raw_float32', 'sample_rate': 16000},
            {'format': 'raw_int16', 'sample_rate': 16000},
            {'format': 'webm', 'sample_rate': 48000}
        ]
        
        for audio_format in audio_formats:
            audio_data = np.random.random(1600).astype(np.float32)
            message = {
                'type': 'audio',
                'data': audio_data.tobytes().hex(),
                **audio_format
            }
            
            await session_handler.handle_message(json.dumps(message))
            
            # Should handle format conversion
            session_handler.stt_service.process_audio_chunk.assert_called()


class TestWebSocketEndpoints:
    """Test WebSocket API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_websocket_connection(self, client):
        """Test WebSocket connection establishment."""
        with client.websocket_connect("/ws/voice/stream/test_session") as websocket:
            # Should establish connection successfully
            assert websocket is not None
    
    def test_websocket_message_exchange(self, client):
        """Test basic message exchange over WebSocket."""
        with patch('backend.app.api.v1.voice_ws.VoiceSessionHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler.initialize_session = AsyncMock()
            mock_handler.handle_message = AsyncMock()
            mock_handler.handle_disconnect = AsyncMock()
            mock_handler_class.return_value = mock_handler
            
            with client.websocket_connect("/ws/voice/stream/test_session") as websocket:
                # Send test message
                test_message = {'type': 'text', 'text': 'Hello'}
                websocket.send_text(json.dumps(test_message))
                
                # Should call handler
                mock_handler.handle_message.assert_called()
    
    def test_websocket_audio_streaming(self, client):
        """Test audio streaming over WebSocket."""
        with patch('backend.app.api.v1.voice_ws.VoiceSessionHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler.initialize_session = AsyncMock()
            mock_handler.handle_message = AsyncMock()
            mock_handler.handle_disconnect = AsyncMock()
            mock_handler_class.return_value = mock_handler
            
            with client.websocket_connect("/ws/voice/stream/audio_test") as websocket:
                # Send audio data
                audio_data = np.random.random(1600).astype(np.float32)
                audio_message = {
                    'type': 'audio',
                    'data': audio_data.tobytes().hex(),
                    'format': 'raw_float32',
                    'sample_rate': 16000
                }
                
                websocket.send_text(json.dumps(audio_message))
                
                # Should process audio
                mock_handler.handle_message.assert_called()
    
    def test_websocket_disconnection_handling(self, client):
        """Test WebSocket disconnection handling."""
        with patch('backend.app.api.v1.voice_ws.VoiceSessionHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler.initialize_session = AsyncMock()
            mock_handler.handle_disconnect = AsyncMock()
            mock_handler_class.return_value = mock_handler
            
            with client.websocket_connect("/ws/voice/stream/disconnect_test") as websocket:
                # Connection established
                pass
            
            # Should handle disconnect
            mock_handler.handle_disconnect.assert_called()
    
    def test_websocket_error_handling(self, client):
        """Test WebSocket error handling."""
        with patch('backend.app.api.v1.voice_ws.VoiceSessionHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler.initialize_session = AsyncMock(side_effect=Exception("Initialization error"))
            mock_handler_class.return_value = mock_handler
            
            # Should handle initialization errors gracefully
            with pytest.raises(Exception):
                with client.websocket_connect("/ws/voice/stream/error_test") as websocket:
                    pass
    
    def test_multiple_concurrent_connections(self, client):
        """Test handling multiple concurrent WebSocket connections."""
        with patch('backend.app.api.v1.voice_ws.VoiceSessionHandler') as mock_handler_class:
            mock_handlers = []
            
            def create_mock_handler(*args, **kwargs):
                handler = Mock()
                handler.initialize_session = AsyncMock()
                handler.handle_message = AsyncMock()
                handler.handle_disconnect = AsyncMock()
                mock_handlers.append(handler)
                return handler
            
            mock_handler_class.side_effect = create_mock_handler
            
            # Create multiple connections
            connections = []
            try:
                for i in range(3):
                    websocket = client.websocket_connect(f"/ws/voice/stream/concurrent_test_{i}")
                    connections.append(websocket.__enter__())
                
                # Should create separate handlers for each connection
                assert len(mock_handlers) == 3
                
                # Send messages to each connection
                for i, websocket in enumerate(connections):
                    message = {'type': 'text', 'text': f'Message {i}'}
                    websocket.send_text(json.dumps(message))
                
                # Each handler should receive its message
                for handler in mock_handlers:
                    handler.handle_message.assert_called()
                    
            finally:
                # Clean up connections
                for websocket in connections:
                    websocket.__exit__(None, None, None)


@pytest.mark.integration
class TestWebSocketIntegration:
    """Integration tests for WebSocket voice streaming."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_voice_session(self):
        """Test complete voice session flow."""
        # This test would require actual service integration
        # For now, we'll test the structure
        
        session_id = "integration_test_session"
        
        # Mock all services
        with patch('ai.voice.stt_service.STTService') as mock_stt, \
             patch('ai.voice.tts_service.TTSService') as mock_tts, \
             patch('ai.conversation.state_manager.ConversationStateManager') as mock_state, \
             patch('ai.conversation.dialog_flow.DialogFlowEngine') as mock_dialog, \
             patch('ai.decision_engine.intent_classifier.IntentClassifier') as mock_intent:
            
            # Configure mocks
            mock_stt_instance = mock_stt.return_value
            mock_stt_instance.create_session.return_value = {'status': 'active'}
            mock_stt_instance.process_audio_chunk.return_value = {'partial_text': 'Hello'}
            mock_stt_instance.get_final_result.return_value = {
                'text': 'I need legal consultation',
                'confidence': 0.95
            }
            
            mock_tts_instance = mock_tts.return_value
            mock_tts_instance.synthesize_async = AsyncMock(return_value={
                'audio': b'response_audio',
                'metadata': {'duration': 2.0}
            })
            
            mock_state_instance = mock_state.return_value
            mock_session = Mock()
            mock_session.add_turn = Mock()
            mock_state_instance.create_session = AsyncMock(return_value=mock_session)
            
            mock_dialog_instance = mock_dialog.return_value
            mock_dialog_instance.start_flow.return_value = {
                'response': 'Hello! How can I help you?',
                'next_action': 'wait_for_input'
            }
            mock_dialog_instance.process_input.return_value = {
                'response': 'I can help with legal consultation.',
                'next_action': 'wait_for_input'
            }
            
            mock_intent_instance = mock_intent.return_value
            mock_intent_instance.classify.return_value = {
                'intent': 'legal_consultation',
                'confidence': 0.9
            }
            
            # Create handler
            handler = VoiceSessionHandler(
                session_id=session_id,
                stt_service=mock_stt_instance,
                tts_service=mock_tts_instance,
                conversation_state=mock_state_instance,
                dialog_engine=mock_dialog_instance,
                intent_classifier=mock_intent_instance
            )
            
            # Mock WebSocket
            mock_websocket = Mock()
            mock_websocket.send_text = AsyncMock()
            mock_websocket.send_bytes = AsyncMock()
            
            # Initialize session
            await handler.initialize_session(mock_websocket)
            
            # Process audio message
            audio_data = np.random.random(16000).astype(np.float32)  # 1 second
            audio_message = {
                'type': 'audio',
                'data': audio_data.tobytes().hex(),
                'format': 'raw_float32',
                'sample_rate': 16000
            }
            
            await handler.handle_message(json.dumps(audio_message))
            
            # Simulate final transcription
            await handler.process_final_transcription()
            
            # Verify the flow
            mock_stt_instance.create_session.assert_called_once()
            mock_state_instance.create_session.assert_called_once()
            mock_intent_instance.classify.assert_called()
            mock_dialog_instance.process_input.assert_called()
            mock_tts_instance.synthesize_async.assert_called()
            
            # Should send responses back
            mock_websocket.send_text.assert_called()
            mock_websocket.send_bytes.assert_called()
    
    @pytest.mark.asyncio
    async def test_websocket_performance(self):
        """Test WebSocket performance under load."""
        import time
        
        # Create multiple handlers for performance testing
        handlers = []
        
        with patch('ai.voice.stt_service.STTService'), \
             patch('ai.voice.tts_service.TTSService'), \
             patch('ai.conversation.state_manager.ConversationStateManager'), \
             patch('ai.conversation.dialog_flow.DialogFlowEngine'), \
             patch('ai.decision_engine.intent_classifier.IntentClassifier'):
            
            start_time = time.time()
            
            # Create multiple handlers
            for i in range(10):
                handler = VoiceSessionHandler(
                    session_id=f"perf_test_{i}",
                    stt_service=Mock(),
                    tts_service=Mock(),
                    conversation_state=Mock(),
                    dialog_engine=Mock(),
                    intent_classifier=Mock()
                )
                handlers.append(handler)
            
            end_time = time.time()
            
            # Should create handlers quickly
            assert len(handlers) == 10
            assert (end_time - start_time) < 1.0  # Should complete quickly