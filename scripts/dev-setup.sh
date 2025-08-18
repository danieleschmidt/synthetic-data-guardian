#!/bin/bash

# Development Environment Setup Script for Synthetic Data Guardian
# This script sets up the complete development environment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Install system dependencies
install_system_deps() {
    local os=$(detect_os)
    log_info "Installing system dependencies for $os..."
    
    case $os in
        "linux")
            if command_exists apt-get; then
                sudo apt-get update
                sudo apt-get install -y git curl build-essential python3 python3-pip python3-venv nodejs npm docker.io docker-compose
            elif command_exists yum; then
                sudo yum update -y
                sudo yum install -y git curl gcc gcc-c++ make python3 python3-pip nodejs npm docker docker-compose
            elif command_exists dnf; then
                sudo dnf update -y
                sudo dnf install -y git curl gcc gcc-c++ make python3 python3-pip nodejs npm docker docker-compose
            else
                log_warning "Unknown package manager. Please install dependencies manually."
            fi
            ;;
        "macos")
            if command_exists brew; then
                brew update
                brew install git curl python3 node docker docker-compose
            else
                log_warning "Homebrew not found. Please install Homebrew first: https://brew.sh/"
            fi
            ;;
        "windows")
            log_warning "Windows detected. Please install dependencies manually or use WSL2."
            ;;
        *)
            log_warning "Unknown OS. Please install dependencies manually."
            ;;
    esac
}

# Setup Node.js environment
setup_node() {
    log_info "Setting up Node.js environment..."
    
    # Check Node.js version
    if command_exists node; then
        local node_version=$(node --version | cut -d'v' -f2)
        local required_version="18.0.0"
        if [[ "$(printf '%s\n' "$required_version" "$node_version" | sort -V | head -n1)" != "$required_version" ]]; then
            log_warning "Node.js version $node_version is below required $required_version"
        else
            log_success "Node.js version $node_version is compatible"
        fi
    else
        log_error "Node.js not found. Please install Node.js 18+"
        return 1
    fi
    
    # Install Node.js dependencies
    log_info "Installing Node.js dependencies..."
    npm ci
    
    # Install global tools
    log_info "Installing global development tools..."
    npm install -g nodemon @playwright/test husky lint-staged
    
    log_success "Node.js environment setup complete"
}

# Setup Python environment
setup_python() {
    log_info "Setting up Python environment..."
    
    # Check Python version
    if command_exists python3; then
        local python_version=$(python3 --version | cut -d' ' -f2)
        local required_version="3.9.0"
        if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
            log_warning "Python version $python_version is below required $required_version"
        else
            log_success "Python version $python_version is compatible"
        fi
    else
        log_error "Python 3 not found. Please install Python 3.9+"
        return 1
    fi
    
    # Create virtual environment
    if [[ ! -d "venv" ]]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment and install dependencies
    log_info "Installing Python dependencies..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    
    log_success "Python environment setup complete"
}

# Setup Docker environment
setup_docker() {
    log_info "Setting up Docker environment..."
    
    if command_exists docker; then
        # Start Docker service if not running
        if ! docker info >/dev/null 2>&1; then
            log_info "Starting Docker service..."
            if command_exists systemctl; then
                sudo systemctl start docker
                sudo systemctl enable docker
            elif command_exists service; then
                sudo service docker start
            else
                log_warning "Please start Docker service manually"
            fi
        fi
        
        # Build development images
        log_info "Building Docker images..."
        docker-compose -f docker-compose.dev.yml build
        
        log_success "Docker environment setup complete"
    else
        log_warning "Docker not found. Please install Docker first."
    fi
}

# Setup pre-commit hooks
setup_precommit() {
    log_info "Setting up pre-commit hooks..."
    
    if command_exists pre-commit; then
        pre-commit install
        pre-commit install --hook-type commit-msg
        log_success "Pre-commit hooks installed"
    else
        log_info "Installing pre-commit..."
        pip install pre-commit
        pre-commit install
        pre-commit install --hook-type commit-msg
        log_success "Pre-commit installed and configured"
    fi
}

# Setup database services
setup_databases() {
    log_info "Setting up database services..."
    
    # Start database services with Docker Compose
    if command_exists docker-compose; then
        docker-compose -f docker-compose.dev.yml up -d postgres redis neo4j
        
        # Wait for services to be ready
        log_info "Waiting for database services to be ready..."
        sleep 10
        
        # Test connections
        if docker-compose -f docker-compose.dev.yml exec -T postgres pg_isready -U postgres; then
            log_success "PostgreSQL is ready"
        else
            log_warning "PostgreSQL may not be ready yet"
        fi
        
        if docker-compose -f docker-compose.dev.yml exec -T redis redis-cli ping | grep -q PONG; then
            log_success "Redis is ready"
        else
            log_warning "Redis may not be ready yet"
        fi
        
        log_success "Database services started"
    else
        log_warning "Docker Compose not found. Database services not started."
    fi
}

# Run initial tests
run_initial_tests() {
    log_info "Running initial tests to verify setup..."
    
    # Run basic tests
    npm run test -- --passWithNoTests
    npm run lint
    npm run typecheck
    
    log_success "Initial tests passed"
}

# Setup IDE configuration
setup_ide() {
    log_info "Setting up IDE configuration..."
    
    # Install VS Code extensions if VS Code is available
    if command_exists code; then
        log_info "Installing VS Code extensions..."
        code --install-extension esbenp.prettier-vscode
        code --install-extension dbaeumer.vscode-eslint
        code --install-extension ms-python.python
        code --install-extension ms-python.black-formatter
        code --install-extension ms-python.flake8
        code --install-extension ms-python.mypy-type-checker
        code --install-extension ms-vscode.vscode-typescript-next
        code --install-extension bradlc.vscode-tailwindcss
        code --install-extension formulahendry.auto-rename-tag
        code --install-extension christian-kohler.path-intellisense
        code --install-extension streetsidesoftware.code-spell-checker
        
        log_success "VS Code extensions installed"
    else
        log_info "VS Code not found. Skipping extension installation."
    fi
}

# Main setup function
main() {
    log_info "Starting Synthetic Data Guardian development environment setup..."
    
    # Check if we're in the right directory
    if [[ ! -f "package.json" ]] || [[ ! -f "requirements.txt" ]]; then
        log_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Parse command line arguments
    SKIP_SYSTEM_DEPS=false
    SKIP_DOCKER=false
    SKIP_DATABASES=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-system-deps)
                SKIP_SYSTEM_DEPS=true
                shift
                ;;
            --skip-docker)
                SKIP_DOCKER=true
                shift
                ;;
            --skip-databases)
                SKIP_DATABASES=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-system-deps    Skip system dependency installation"
                echo "  --skip-docker         Skip Docker setup"
                echo "  --skip-databases      Skip database service setup"
                echo "  --help                Show this help message"
                exit 0
                ;;
            *)
                log_warning "Unknown option: $1"
                shift
                ;;
        esac
    done
    
    # Run setup steps
    if [[ "$SKIP_SYSTEM_DEPS" != "true" ]]; then
        install_system_deps
    fi
    
    setup_node
    setup_python
    
    if [[ "$SKIP_DOCKER" != "true" ]]; then
        setup_docker
    fi
    
    setup_precommit
    
    if [[ "$SKIP_DATABASES" != "true" ]]; then
        setup_databases
    fi
    
    setup_ide
    run_initial_tests
    
    log_success "Development environment setup complete!"
    log_info "You can now start development with:"
    log_info "  npm run dev          # Start development server"
    log_info "  npm run test:watch   # Run tests in watch mode"
    log_info "  docker-compose -f docker-compose.dev.yml up -d  # Start all services"
}

# Run main function
main "$@"