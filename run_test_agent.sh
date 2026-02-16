#!/bin/bash

# Quick Test Script for LiveBench Dashboard
# Runs an agent with specified config to populate the dashboard
#
# Usage:
#   ./run_test_agent.sh                              # Uses default config
#   ./run_test_agent.sh livebench/configs/test_glm47.json

# Get config file from argument or use default
CONFIG_FILE=${1:-"livebench/configs/test_gpt4o.json"}

echo "🎯 LiveBench Agent Test"
echo "===================================="
echo ""
echo "📋 Config: $CONFIG_FILE"
echo ""

# Activate conda environment
echo "🔧 Activating livebench conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate livebench
echo "   Using Python: $(which python)"
echo ""

# Load environment variables from .env if it exists
if [ -f ".env" ]; then
    echo "📝 Loading environment variables from .env..."
    source .env
    echo ""
fi

# Validate config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    echo ""
    echo "Available configs:"
    ls -1 livebench/configs/*.json 2>/dev/null || echo "  (none found)"
    echo ""
    exit 1
fi
echo "✓ Config file found"
echo ""

# Check environment variables
echo "🔍 Checking environment..."

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not set"
    echo "   Please set it: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi
echo "✓ OPENAI_API_KEY set"

if [ -z "$WEB_SEARCH_API_KEY" ]; then
    echo "❌ WEB_SEARCH_API_KEY not set"
    echo "   Please set it: export WEB_SEARCH_API_KEY='your-key-here'"
    echo "   You can also set WEB_SEARCH_PROVIDER (default: tavily)"
    exit 1
fi
echo "✓ WEB_SEARCH_API_KEY set"

echo ""

# Set MCP port if not set
export LIVEBENCH_HTTP_PORT=${LIVEBENCH_HTTP_PORT:-8010}

# Add project root to PYTHONPATH to ensure imports work
export PYTHONPATH="/root/-Live-Bench:$PYTHONPATH"

# Extract agent info from config (basic parsing)
AGENT_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('signature', 'unknown'))")
BASEMODEL=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('basemodel', 'unknown'))")
INIT_DATE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('init_date', 'N/A'))")
END_DATE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('end_date', 'N/A'))")
INITIAL_BALANCE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('initial_balance', 1000))")

echo "===================================="
echo "🤖 Running Agent"
echo "===================================="
echo ""
echo "Configuration:"
echo "  - Config: $(basename $CONFIG_FILE)"
echo "  - Agent: ${AGENT_NAME:-unknown}"
echo "  - Model: ${BASEMODEL:-unknown}"
echo "  - Date Range: ${INIT_DATE:-N/A} to ${END_DATE:-N/A}"
echo "  - Initial Balance: \$${INITIAL_BALANCE:-1000}"
echo ""
echo "Note: The agent will handle MCP service internally"
echo ""
echo "This will take a few minutes..."
echo ""
echo "===================================="
echo ""

# Run the agent with specified config
python livebench/main.py "$CONFIG_FILE"

echo ""
echo "===================================="
echo "✅ Test completed!"
echo "===================================="
echo ""
echo "📊 View results in dashboard:"
echo "   http://localhost:3000"
echo ""
echo "🔧 API endpoints:"
echo "   http://localhost:8000/api/agents"
echo "   http://localhost:8000/docs"
echo ""
