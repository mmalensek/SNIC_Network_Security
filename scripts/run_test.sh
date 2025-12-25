#!/usr/bin/env bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp once for both files
ts="$(date +%Y-%m-%d_%H-%M)"

# Run ModelTester with logging
echo "Running ModelTester.py..."
python ModelTester.py 2>&1 | tee "logs/testing_${ts}.txt"

# Run txt2md on the generated log file
echo "Converting log to markdown..."
python txt2md.py "logs/testing_${ts}.txt"

echo "Done! Check logs/testing_${ts}.txt and its .md version."[web:26][web:31]
