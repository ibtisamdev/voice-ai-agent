# Future Enhancements

This document tracks planned features and improvements for the Voice AI Agent project.

## Chat Interface for Testing

### Overview
Create a web-based chat interface for testing the AI agent with both text and voice input options.

### Requirements

#### Core Features
- **Dual Input Modes**:
  - Text input field for quick testing
  - Voice recording button for audio testing
  - Toggle between modes seamlessly

- **Message History**:
  - Display conversation history (user/AI messages)
  - Show timestamps
  - Visual distinction between user and AI messages
  - Auto-scroll to latest message

- **Debug Panel** (collapsible/toggleable):
  - **Intent Detection**: Show detected intent + confidence score
  - **Slot Collection**: Display collected slots and their values
  - **Flow State**: Current flow node and execution path
  - **Session Variables**: Real-time view of session variables
  - **Transcription**: Show STT output when using voice

- **Session Controls**:
  - Reset conversation
  - Change conversation flow (dropdown selector)
  - End session manually
  - Session ID display

- **Audio Playback**:
  - Play TTS responses inline
  - Audio waveform visualization (optional)
  - Download audio option

#### Technical Implementation

##### Frontend Options
1. **Simple HTML/JS** (Quick prototyping):
   - Single file application
   - WebSocket connection to `/ws/voice/stream/{session_id}`
   - Minimal dependencies
   - Fast to implement and deploy

2. **React Application** (Production-ready):
   - Better UX with component reusability
   - State management (Redux/Context)
   - Professional UI (Material-UI/Tailwind)
   - TypeScript for type safety

3. **Both approaches**:
   - HTML for quick developer tests
   - React for demos and client presentations

##### Backend Requirements
- WebSocket handler already supports text input (voice_ws.py:356-360)
- Add new endpoint to serve chat UI: `GET /chat` or `GET /test`
- Static file serving for HTML/JS/React build
- No additional API changes needed

##### Architecture
```
Frontend (React/HTML)
    ↓
WebSocket /ws/voice/stream/{session_id}
    ↓
    ├─ Text messages: {"type": "text", "text": "..."}
    └─ Audio messages: {"type": "audio", "data": "base64..."}
    ↓
Backend (existing pipeline)
    ↓
Response: {"type": "response", "text": "...", "audio": "base64..."}
```

##### File Structure
```
frontend/
├── chat-test/              # Simple HTML version
│   ├── index.html
│   ├── chat.js
│   └── styles.css
│
├── chat-ui/                # React version
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatWindow.tsx
│   │   │   ├── MessageList.tsx
│   │   │   ├── InputPanel.tsx
│   │   │   ├── DebugPanel.tsx
│   │   │   └── SessionControls.tsx
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts
│   │   │   └── useAudioRecorder.ts
│   │   ├── types/
│   │   │   └── messages.ts
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts
```

##### Backend Changes Needed
```python
# backend/app/main.py - Add static file serving
from fastapi.staticfiles import StaticFiles

app.mount("/chat", StaticFiles(directory="frontend/chat-test", html=True), name="chat")
app.mount("/chat-ui", StaticFiles(directory="frontend/chat-ui/dist", html=True), name="chat-ui")
```

#### Benefits
1. **Development Speed**: Test conversation flows without recording audio every time
2. **Debug Visibility**: See exactly what the AI is thinking (intents, slots, state)
3. **Flow Iteration**: Quickly test different conversation paths
4. **Voice Pipeline Testing**: Test STT/TTS independently when needed
5. **Demo Tool**: Show AI capabilities to stakeholders
6. **Documentation**: Visual guide for how conversations flow

#### Priority
- **Phase**: Post-Phase 4 (after production deployment)
- **Effort**: 2-3 days for HTML version, 1 week for React version
- **Dependencies**: None (WebSocket already supports text)

#### Notes
- Can be built incrementally (HTML first, then React)
- Useful for both development and client demonstrations
- Consider adding conversation export (JSON/CSV) for analysis
- Could evolve into admin panel for flow management
