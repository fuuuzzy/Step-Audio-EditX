#!/bin/bash

# Install system dependencies
echo "Installing system dependencies..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        sudo apt update
        sudo apt install -y ffmpeg sox
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        sudo yum install -y ffmpeg sox
    elif command -v pacman &> /dev/null; then
        # Arch Linux
        sudo pacman -S --noconfirm ffmpeg sox
    else
        echo "Warning: Could not detect package manager. Please install ffmpeg and sox manually."
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if command -v brew &> /dev/null; then
        brew install ffmpeg sox
    else
        echo "Warning: Homebrew not found. Please install ffmpeg and sox manually:"
        echo "  brew install ffmpeg sox"
    fi
else
    echo "Warning: Unsupported OS. Please install ffmpeg and sox manually."
fi

# Install Python dependencies
echo "Installing Python dependencies..."
uv sync

# Install optional dependencies if needed
# Uncomment the following lines if you want to install optional dependencies by default:
# echo "Installing optional dependencies..."
# uv pip install hdbscan rotary-embedding-torch ffmpeg-python

# Setup git lfs
git lfs install

# Download models
echo "Downloading models..."
mkdir -p models
git clone https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer models/Step-Audio-Tokenizer
git clone https://huggingface.co/stepfun-ai/Step-Audio-EditX models/Step-Audio-EditX