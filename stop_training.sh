#!/bin/bash

ps aux | grep "python src/train_model.py" | grep -v "grep" | awk '{print $2}' | xargs kill -9
echo "training stopped..."
