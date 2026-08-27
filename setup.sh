#!/usr/bin/env bash
# One-shot project setup for Ubuntu -- works unchanged in WSL2 and on an
# AWS EC2 Ubuntu instance. Run system deps (step 5 in the README) first,
# then run this from inside the project folder:
#
#   bash setup.sh
set -euo pipefail

echo "Creating virtual environment (./venv)..."
python3 -m venv venv

echo "Installing Python requirements..."
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

echo ""
echo "Done. Activate the environment with:"
echo "    source venv/bin/activate"
echo ""
echo "Then:"
echo "    python -m stable_retro.import /path/to/your/ROMs/"
echo "    python list_games.py"
echo "    python play_and_record.py --game SuperMarioBros-Nes"
