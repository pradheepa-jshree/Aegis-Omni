#!/bin/bash
# run_all.sh — Full Aegis-Omni stack launcher
# Usage: bash run_all.sh [--synthetic]

set -e
BOLD='\033[1m'; CYAN='\033[96m'; GREEN='\033[92m'; RESET='\033[0m'

echo -e "${BOLD}${CYAN}"
echo "  ╔═══════════════════════════════════════╗"
echo "  ║   AEGIS-OMNI — Full Stack Launcher    ║"
echo "  ╚═══════════════════════════════════════╝"
echo -e "${RESET}"

# Step 1: Install dependencies
echo -e "${BOLD}Installing dependencies...${RESET}"
pip install -r requirements.txt -q

# Step 2: Train (if artifacts don't exist)
if [ ! -f "artifacts/best_model.pt" ]; then
    echo -e "${BOLD}No trained model found — running setup_and_train.py...${RESET}"
    python setup_and_train.py "$@"
else
    echo -e "${GREEN}✓ Trained model found — skipping training${RESET}"
fi

# Step 3: Launch API in background
echo -e "${BOLD}Starting FastAPI backend on :8000...${RESET}"
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!
sleep 3

# Health check
STATUS=$(curl -s http://localhost:8000/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))")
echo -e "${GREEN}✓ API status: $STATUS${RESET}"

# Step 4: Launch Streamlit
echo -e "${BOLD}Starting Streamlit dashboard on :8501...${RESET}"
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
DASH_PID=$!

echo ""
echo -e "${GREEN}${BOLD}  ✅ Aegis-Omni is running!${RESET}"
echo ""
echo "  API:       http://localhost:8000"
echo "  API docs:  http://localhost:8000/docs"
echo "  Dashboard: http://localhost:8501"
echo ""
echo "  Ctrl+C to stop all services"

# Wait and cleanup on exit
trap "echo 'Shutting down...'; kill $API_PID $DASH_PID 2>/dev/null" EXIT
wait
