#!/bin/bash

# Interview Analysis System - Complete Pipeline Runner
# Click this file in VS Code and run it to execute the entire analysis process

set -e  # Exit on any error

echo "🚀 Starting Interview Analysis Pipeline..."
echo "========================================"

# Check if .env exists and is configured
if [ ! -f .env ]; then
    echo "❌ .env file not found. Copying template..."
    cp .env.template .env
    echo "⚠️  Please edit .env with your OpenAI API key before running again!"
    exit 1
fi

if grep -q "your_openai_api_key_here" .env; then
    echo "⚠️  Please set your OPENAI_API_KEY in .env file before running!"
    exit 1
fi

echo "✓ Environment configured"
echo ""

# Run the complete analysis pipeline
echo "📊 Running complete analysis pipeline..."
make run

echo ""
echo "🎉 Analysis pipeline completed successfully!"
echo "📁 Check the results/ directory for output files"