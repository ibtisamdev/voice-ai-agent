"""
Celery configuration for Voice AI Agent.
Handles asynchronous tasks for voice processing, document analysis, and background operations.
"""

from celery import Celery
from .config import settings

# Create Celery instance
celery_app = Celery(
    "voiceai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=['app.tasks']
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task routing and execution
    task_routes={
        'app.tasks.voice.*': {'queue': 'voice'},
        'app.tasks.document.*': {'queue': 'documents'},
        'app.tasks.ai.*': {'queue': 'ai_processing'},
    },
    
    # Task result settings
    result_expires=3600,  # 1 hour
    task_track_started=True,
    task_time_limit=300,  # 5 minutes
    task_soft_time_limit=240,  # 4 minutes
    
    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Periodic tasks (optional - for future scheduled tasks)
    beat_schedule={},
    beat_schedule_filename='/app/celerybeat-data/schedule',
)

# Make celery instance available for import
__all__ = ['celery_app']