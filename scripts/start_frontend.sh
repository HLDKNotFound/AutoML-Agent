#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../frontend"
echo "Starting ML Automation Agent React Frontend on http://localhost:5173 ..."
npm run dev -- --host 0.0.0.0 --port 5173
