#!/bin/bash

# Wrapper script for Automator app - provides better user experience
# This script is called by the Automator application

set -e  # Exit on any error

# Function to show dialog on macOS
show_dialog() {
    osascript -e "display dialog \"$1\" buttons {\"OK\"} default button \"OK\""
}

# Function to show notification
show_notification() {
    osascript -e "display notification \"$1\" with title \"Analysis Complete\""
}

echo "🚀 Interview Analysis Pipeline Starting..."
echo "========================================"

# Check if we're in the right directory
if [ ! -f "Makefile" ]; then
    echo "❌ Error: Not in the correct project directory"
    show_dialog "Error: Could not find project files. Please ensure the app is properly configured."
    exit 1
fi

# Check if .env exists and is configured
if [ ! -f .env ]; then
    echo "❌ .env file not found. Copying template..."
    cp .env.template .env
    show_dialog "Setup needed: Please edit the .env file with your OpenAI API key, then run the app again."
    open .env
    exit 1
fi

if grep -q "your_openai_api_key_here" .env; then
    show_dialog "Setup needed: Please set your OPENAI_API_KEY in the .env file, then run the app again."
    open .env
    exit 1
fi

echo "✓ Environment configured"
echo ""

# Run the complete analysis pipeline
echo "📊 Running complete analysis pipeline..."
echo "This may take several minutes..."

if make run; then
    echo ""
    echo "🎉 Analysis pipeline completed successfully!"
    echo "📁 Check the results/ directory for output files"
    show_notification "Analysis completed successfully! Check the results folder."

    # Ask if user wants to open results folder
    if osascript -e 'display dialog "Analysis completed successfully! Would you like to open the results folder?" buttons {"No", "Yes"} default button "Yes"' | grep -q "Yes"; then
        open results/
    fi
else
    echo ""
    echo "❌ Analysis pipeline failed!"
    show_dialog "Analysis failed. Please check the terminal output for error details."
    exit 1
fi

# Keep terminal open so user can see the output
echo ""
echo "Press any key to close this window..."
read -n 1 -s