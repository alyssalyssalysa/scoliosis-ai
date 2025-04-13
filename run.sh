#!/bin/bash

PORT=5001
echo "🔍 Checking port $PORT..."

# Find and kill any processes using the port
PIDS=$(lsof -ti tcp:$PORT)
if [ -n "$PIDS" ]; then
  echo "⚠️ Port $PORT is in use by PID(s): $PIDS"
  kill -9 $PIDS
  echo "✅ Killed processes using port $PORT"
else
  echo "✅ Port $PORT is free"
fi

# Start the app
echo "🚀 Starting app on port $PORT..."
python3 app.py --port $PORT

