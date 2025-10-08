# Voice AI Agent - Jupyter Notebooks

Interactive notebooks for data analysis and experimentation with the Voice AI system.

## Quick Start

### 1. Install Jupyter Locally

Using `uv` (recommended - much faster):

```bash
cd notebooks/
uv pip install -r requirements.txt
```

Or using regular pip:

```bash
cd notebooks/
pip install -r requirements.txt
```

### 2. Start Jupyter Lab

```bash
jupyter lab
```

Jupyter Lab will open in your browser automatically. If not, navigate to the URL shown in the terminal (usually `http://localhost:8888`).

### 3. Open a Notebook

In Jupyter Lab, navigate to one of the notebook directories:

- **`demos/`** - Getting started demos
- **`data_analysis/`** - Analytics and data exploration
- **`model_testing/`** - AI model testing
- **`experiments/`** - Your custom experiments

### 4. VS Code Support

The notebooks work great in VS Code! Just:
1. Open the notebook file (`.ipynb`)
2. Select your Python kernel when prompted
3. Run cells with Shift+Enter

## Available Notebooks

### 📘 Demos

#### `demos/01_voice_pipeline_demo.ipynb`
Introduction to querying Voice AI data. Learn how to:
- Connect to PostgreSQL and Redis
- Query conversation data
- Visualize session statistics
- Test API endpoints

**Best for**: Getting started, understanding data structures

### 📊 Data Analysis

#### `data_analysis/conversation_analytics.ipynb`
Comprehensive conversation analytics. Includes:
- Session statistics and trends
- Conversation turn analysis
- Intent distribution analysis
- Daily/hourly analytics summaries
- Custom query examples

**Best for**: Understanding usage patterns, creating reports

### 🧪 Model Testing

#### `model_testing/whisper_testing.ipynb`
Analyze transcription data and performance:
- STT accuracy metrics
- Processing time analysis
- Confidence score distributions
- Error pattern identification

**Best for**: Evaluating STT quality, identifying issues

## Connecting to Services

Your Docker services expose ports to localhost, making them accessible from local Jupyter:

| Service | Docker Port | Connection String |
|---------|-------------|-------------------|
| PostgreSQL | 5432 | `postgresql://voiceai:voiceai_dev@localhost:5432/voiceai_db` |
| Redis | 6379 | `redis://localhost:6379` |
| API | 8000 | `http://localhost:8000` |
| ChromaDB | 8001 | `http://localhost:8001` |
| Ollama | 11434 | `http://localhost:11434` |

### Example: Connect to PostgreSQL

```python
import pandas as pd
from sqlalchemy import create_engine

# Create database connection
engine = create_engine('postgresql://voiceai:voiceai_dev@localhost:5432/voiceai_db')

# Query data
query = """
SELECT session_id, status, created_at
FROM conversations.conversation_sessions
LIMIT 10;
"""

df = pd.read_sql(query, engine)
print(df)
```

### Example: Connect to Redis

```python
import redis

# Connect to Redis
r = redis.from_url('redis://localhost:6379')

# Test connection
r.ping()  # Returns True

# Get session keys
keys = r.keys('conversation:session:*')
print(f"Active sessions: {len(keys)}")
```

### Example: Call API Endpoints

```python
import requests

# Test health endpoint
response = requests.get('http://localhost:8000/api/v1/health')
print(response.json())

# Test voice synthesis
response = requests.post(
    'http://localhost:8000/api/v1/voice/synthesize',
    json={'text': 'Hello world', 'voice_id': 'default'}
)

if response.ok:
    with open('test_audio.wav', 'wb') as f:
        f.write(response.content)
    print("Audio saved to test_audio.wav")
```

## Database Schemas

### Conversations Schema

- `conversation_sessions` - Session metadata (status, type, timestamps)
- `conversation_turns` - Individual turns (user input, bot response, intent)
- `conversation_context` - Session context and state

### Voice Data Schema

- `transcriptions` - STT results and performance metrics
- `syntheses` - TTS results and character counts

### Analytics Schema

- `daily_stats` - Aggregated daily metrics
- `session_summary` - Per-session summaries

Run this query to explore tables:

```sql
-- List all tables in conversations schema
SELECT tablename FROM pg_tables WHERE schemaname = 'conversations';
```

## Tips and Best Practices

### Data Analysis

- **Use pandas** for data manipulation:
  ```python
  df.groupby('session_type').agg({
      'session_id': 'count',
      'duration_seconds': 'mean'
  })
  ```

- **Visualize with matplotlib/seaborn**:
  ```python
  import seaborn as sns
  import matplotlib.pyplot as plt

  sns.histplot(df['duration_seconds'])
  plt.title('Session Duration Distribution')
  plt.show()
  ```

### Working with Time Series

```python
# Convert to datetime
df['created_at'] = pd.to_datetime(df['created_at'])

# Set as index for time series analysis
df.set_index('created_at', inplace=True)

# Resample by day
daily_counts = df.resample('D').size()
daily_counts.plot()
```

### Efficient Queries

- **Use LIMIT** to avoid loading huge datasets
- **Filter with WHERE** to reduce data transfer
- **Aggregate in SQL** rather than loading all rows

```python
# Good - aggregate in database
query = """
SELECT
    DATE(created_at) as date,
    COUNT(*) as sessions,
    AVG(duration_seconds) as avg_duration
FROM conversations.conversation_sessions
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE(created_at);
"""

# Better than loading all sessions and grouping in pandas
```

## Creating Your Own Notebooks

### 1. Create a new notebook

In Jupyter Lab: **File > New > Notebook** or place `.ipynb` files in `experiments/`

### 2. Standard imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import redis

# Database connection
engine = create_engine('postgresql://voiceai:voiceai_dev@localhost:5432/voiceai_db')
```

### 3. Query and analyze

```python
# Load data
query = "SELECT * FROM conversations.conversation_sessions LIMIT 100;"
df = pd.read_sql(query, engine)

# Analyze
print(df.describe())
print(df['status'].value_counts())

# Visualize
df['status'].value_counts().plot(kind='bar')
plt.title('Sessions by Status')
plt.show()
```

## Common Issues

### Issue: Connection refused to localhost:5432

**Solution**: Make sure Docker services are running:
```bash
docker-compose -f docker/docker-compose.yml ps
```

If not running:
```bash
make dev
```

### Issue: Authentication failed for PostgreSQL

**Solution**: Check credentials match those in `docker/docker-compose.yml`:
- User: `voiceai`
- Password: `voiceai_dev`
- Database: `voiceai_db`

### Issue: Redis connection error

**Solution**: Verify Redis is running:
```bash
docker-compose -f docker/docker-compose.yml ps redis
```

### Issue: Module not found (e.g., pandas, matplotlib)

**Solution**: Install notebook requirements:
```bash
cd notebooks/
uv pip install -r requirements.txt
# or: pip install -r requirements.txt
```

## Installed Packages

Core tools:
- `jupyter`, `jupyterlab` - Notebook environment
- `pandas`, `numpy` - Data analysis
- `matplotlib`, `seaborn` - Visualization
- `sqlalchemy`, `psycopg2-binary` - Database access
- `redis` - Redis client
- `requests`, `httpx` - HTTP requests
- `librosa`, `soundfile` - Audio analysis (optional)

## Additional Resources

- [Project Documentation](../README.md)
- [CLAUDE.md](../CLAUDE.md) - Development guide
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [SQLAlchemy Tutorial](https://docs.sqlalchemy.org/en/20/tutorial/)

## Note on AI Module Imports

These notebooks focus on **data analysis** using SQL queries and visualizations. If you need to test AI services directly (STT, TTS, conversation engine), use the API endpoints via HTTP requests or the WebSocket interface.

For direct Python module access to `ai/` and `backend/` code, you would need to add the project to your PYTHONPATH:

```bash
export PYTHONPATH="/Users/ibtisam/Documents/voice-ai-agent:$PYTHONPATH"
```

However, most analysis can be done through database queries and API calls.

---

**Questions or issues?** Check the main project README or open an issue on GitHub.

Happy analyzing! 📊🚀
