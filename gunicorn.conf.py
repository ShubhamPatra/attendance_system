"""Gunicorn configuration for AutoAttendance."""

# Server socket
bind = "0.0.0.0:5000"

# Worker processes
workers = 2
worker_class = "sync"

# Timeout
timeout = 120
graceful_timeout = 30

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "autoattendance"

# Preload app for faster worker startup
preload_app = False

# Max requests before worker restart (prevent memory leaks)
max_requests = 1000
max_requests_jitter = 50
