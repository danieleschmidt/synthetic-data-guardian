#!/bin/bash
set -e

echo "🚀 Running post-create setup for Synthetic Data Guardian..."

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
if [ -f "requirements-dev.txt" ]; then
    pip install -r requirements-dev.txt
fi

if [ -f "pyproject.toml" ]; then
    pip install -e .
fi

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    pre-commit install --hook-type commit-msg
fi

# Setup Git configuration
echo "🔧 Configuring Git..."
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf input

# Create necessary directories
echo "📁 Creating development directories..."
mkdir -p .devcontainer/cache
mkdir -p .devcontainer/extensions
mkdir -p data/temp
mkdir -p logs
mkdir -p coverage

# Set up environment file from template
echo "⚙️ Setting up environment configuration..."
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "📝 Created .env file from .env.example template"
fi

# Install additional development tools
echo "🛠️ Installing additional development tools..."

# Install k6 for performance testing
if ! command -v k6 &> /dev/null; then
    echo "📊 Installing k6 for performance testing..."
    curl https://github.com/grafana/k6/releases/download/v0.47.0/k6-v0.47.0-linux-amd64.tar.gz -L | tar xvz --strip-components 1
    sudo mv k6 /usr/local/bin/
fi

# Install terraform for infrastructure
if ! command -v terraform &> /dev/null; then
    echo "🏗️ Installing Terraform..."
    wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
    sudo apt update && sudo apt install terraform
fi

# Setup shell aliases and functions
echo "🐚 Setting up development aliases..."
cat >> ~/.bashrc << 'EOF'

# Synthetic Data Guardian Development Aliases
alias sgs='npm start'
alias sgd='npm run dev'
alias sgt='npm test'
alias sgta='npm run test:all'
alias sgl='npm run lint'
alias sgb='npm run build'
alias sgc='npm run clean'
alias sgdc='docker-compose up -d'
alias sgdd='docker-compose down'
alias sgdr='docker-compose restart'

# Python aliases
alias py='python'
alias pytest='python -m pytest'
alias black='python -m black'
alias flake8='python -m flake8'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline'
alias gd='git diff'

# Testing shortcuts
alias unit='npm run test'
alias integration='npm run test:integration'
alias e2e='npm run test:e2e'
alias perf='npm run test:performance'

# Development utilities
sg_health() {
    echo "🔍 Checking Synthetic Guardian health..."
    curl -s http://localhost:8080/health | jq .
}

sg_logs() {
    echo "📋 Showing application logs..."
    docker-compose logs -f app
}

sg_reset() {
    echo "🔄 Resetting development environment..."
    docker-compose down -v
    docker-compose up -d
    npm run dev
}

EOF

# Install VS Code extensions if not in container
if [ -z "$DEVCONTAINER" ]; then
    echo "📚 Installing recommended VS Code extensions..."
    code --install-extension ms-python.python
    code --install-extension esbenp.prettier-vscode
    code --install-extension ms-vscode.vscode-eslint
    code --install-extension ms-playwright.playwright
    code --install-extension github.copilot
fi

# Generate development certificates for HTTPS
echo "🔐 Generating development certificates..."
mkdir -p certs
if [ ! -f "certs/dev-cert.pem" ]; then
    openssl req -x509 -newkey rsa:4096 -keyout certs/dev-key.pem -out certs/dev-cert.pem -days 365 -nodes -subj "/C=US/ST=Development/L=Local/O=SyntheticGuardian/CN=localhost"
fi

# Setup database if needed
echo "💾 Checking database setup..."
if command -v docker-compose &> /dev/null; then
    docker-compose up -d db redis neo4j
    echo "⏳ Waiting for databases to be ready..."
    sleep 10
fi

echo "✅ Post-create setup complete!"
echo "🎉 Welcome to Synthetic Data Guardian development environment!"
echo ""
echo "Quick start commands:"
echo "  npm run dev          - Start development server"
echo "  npm test            - Run unit tests"
echo "  npm run test:all    - Run all tests"
echo "  docker-compose up   - Start all services"
echo ""
echo "Development URLs:"
echo "  API:        http://localhost:8080"
echo "  Frontend:   http://localhost:3000"
echo "  Grafana:    http://localhost:3001"
echo "  Prometheus: http://localhost:9090"
echo ""