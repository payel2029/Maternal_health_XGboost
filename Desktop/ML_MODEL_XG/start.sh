#!/bin/bash
# Use uvicorn worker for ASGI (FastAPI)
exec gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT