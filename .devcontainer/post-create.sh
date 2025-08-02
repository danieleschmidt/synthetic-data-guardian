#!/bin/bash

# Synthetic Data Guardian - Post-Create Development Setup
set -e

echo "🚀 Setting up Synthetic Data Guardian development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install additional development tools
echo "🛠️ Installing development tools..."
sudo apt-get install -y \
    curl \
    wget \
    git \
    vim \
    htop \
    tree \
    jq \
    unzip \
    build-essential \
    pkg-config \
    libssl-dev \
    libffi-dev \
    python3-dev

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
if [ -f "requirements-dev.txt" ]; then
    pip install -r requirements-dev.txt
else
    echo "⚠️ requirements-dev.txt not found, installing basic packages..."
    pip install pytest black flake8 isort pre-commit
fi

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
if [ -f "package.json" ]; then
    npm install
else
    echo "⚠️ package.json not found, skipping npm install"
fi

# Setup pre-commit hooks
echo "🔗 Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    pre-commit install --hook-type commit-msg
else
    echo "⚠️ pre-commit not available, skipping hook installation"
fi

# Create local directories
echo "📁 Creating local directories..."
mkdir -p data/input data/output data/temp
mkdir -p logs
mkdir -p .cache

# Setup git configuration
echo "⚙️ Configuring git..."
git config --global pull.rebase false
git config --global init.defaultBranch main
git config --global core.autocrlf input

# Create useful aliases
echo "🔧 Setting up useful aliases..."
cat >> ~/.bashrc << 'EOF'
# Synthetic Data Guardian aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias pytest-cov='pytest --cov=src --cov-report=html --cov-report=term'
alias lint='flake8 src tests && black --check src tests && isort --check src tests'
alias format='black src tests && isort src tests'
alias test-watch='pytest -f'
alias docker-logs='docker-compose logs -f'
alias guardian-dev='python -m synthetic_guardian.cli'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline'

# Docker aliases
alias dc='docker-compose'
alias dcu='docker-compose up'
alias dcd='docker-compose down'
alias dcb='docker-compose build'
EOF

# Source the new aliases
source ~/.bashrc

# Create useful VS Code snippets
echo "✨ Creating VS Code snippets..."
mkdir -p .vscode
cat > .vscode/snippets.json << 'EOF'
{
  "Guardian Pipeline": {
    "scope": "python",
    "prefix": "guardian-pipeline",
    "body": [
      "from synthetic_guardian import Guardian, GenerationPipeline",
      "",
      "# Initialize guardian",
      "guardian = Guardian()",
      "",
      "# Create generation pipeline",
      "pipeline = GenerationPipeline(",
      "    name=\"${1:pipeline_name}\",",
      "    description=\"${2:Pipeline description}\"",
      ")",
      "",
      "# Configure generation",
      "pipeline.configure(",
      "    generator=\"${3:sdv}\",",
      "    data_type=\"${4:tabular}\",",
      "    schema={",
      "        \"${5:column_name}\": \"${6:column_type}\"",
      "    }",
      ")",
      "",
      "# Add validation steps",
      "pipeline.add_validator(\"statistical_fidelity\", threshold=0.95)",
      "pipeline.add_validator(\"privacy_preservation\", epsilon=1.0)",
      "",
      "# Generate data",
      "result = guardian.generate(",
      "    pipeline=pipeline,",
      "    num_records=${7:1000},",
      "    seed=42",
      ")"
    ],
    "description": "Create a basic Guardian pipeline"
  }
}
EOF

# Setup environment variables
echo "🌍 Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "📝 Created .env file from .env.example - please update with your values"
fi

# Run initial health checks
echo "🏥 Running health checks..."
python --version
node --version
npm --version

# Test import of main package (if available)
if python -c "import synthetic_guardian" 2>/dev/null; then
    echo "✅ synthetic_guardian package imported successfully"
else
    echo "⚠️ synthetic_guardian package not yet available"
fi

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Update .env file with your configuration"
echo "   2. Run 'make test' to verify setup"
echo "   3. Run 'make dev' to start development server"
echo "   4. Visit the docs at http://localhost:8080/docs"
echo ""
echo "🛠️ Useful commands:"
echo "   - make help       : Show all available commands"
echo "   - pytest-cov     : Run tests with coverage"
echo "   - lint           : Check code style"
echo "   - format         : Format code"
echo "   - docker-logs    : View container logs"
echo ""