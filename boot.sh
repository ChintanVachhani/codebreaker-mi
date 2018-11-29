#!/bin/bash
source venv/bin/activate
flask translate compile
exec gunicorn -b :8080 --access-logfile - --error-logfile - application