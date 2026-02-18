

# celery -A src.celery.app worker -l info
celery -A src.celery.app.celery_app worker -l info --pool=solo --concurrency=1
