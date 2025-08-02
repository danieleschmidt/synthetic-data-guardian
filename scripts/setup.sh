#!/bin/bash

# =============================================================================
# Synthetic Data Guardian - Development Environment Setup Script
# =============================================================================

set -e

# Colors for output
CYAN='\033[36m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
RESET='\033[0m'

# Configuration
PROJECT_NAME="Synthetic Data Guardian"
PYTHON_VERSION="3.9"
NODE_VERSION="18"

# Functions
print_header() {
    echo -e "${CYAN}===============================================================================${RESET}"
    echo -e "${CYAN}$1${RESET}"
    echo -e "${CYAN}===============================================================================${RESET}"
    echo
}

print_info() {
    echo -e "${GREEN}[INFO]${RESET} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${RESET} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${RESET} $1"
}

check_command() {
    if command -v $1 &> /dev/null; then
        print_info "$1 is installed"
        return 0
    else
        print_warning "$1 is not installed"
        return 1
    fi
}

install_dependencies() {
    print_header "Installing Dependencies"
    
    # Check for required tools
    print_info "Checking for required tools..."
    
    MISSING_TOOLS=()
    
    if ! check_command "python3"; then
        MISSING_TOOLS+=("python3")
    fi
    
    if ! check_command "pip3"; then
        MISSING_TOOLS+=("pip3")
    fi
    
    if ! check_command "node"; then
        MISSING_TOOLS+=("node")
    fi
    
    if ! check_command "npm"; then
        MISSING_TOOLS+=("npm")
    fi
    
    if ! check_command "docker"; then
        MISSING_TOOLS+=("docker")
    fi
    
    if ! check_command "docker-compose"; then
        MISSING_TOOLS+=("docker-compose")
    fi
    
    if [ ${#MISSING_TOOLS[@]} -ne 0 ]; then
        print_error "Missing required tools: ${MISSING_TOOLS[*]}"
        print_info "Please install the missing tools and run this script again"
        exit 1
    fi
    
    # Install Python dependencies
    print_info "Installing Python dependencies..."
    pip3 install -r requirements-dev.txt
    pip3 install -e ".[dev]"
    
    # Install Node.js dependencies
    print_info "Installing Node.js dependencies..."
    npm install
    
    # Install pre-commit hooks
    print_info "Installing pre-commit hooks..."
    pre-commit install
    
    # Install Husky hooks
    print_info "Installing Husky hooks..."
    npx husky install
    
    print_info "Dependencies installed successfully!"
}

setup_environment() {
    print_header "Setting Up Environment"
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        print_info "Creating .env file from template..."
        cp .env.example .env
        print_warning "Please update .env file with your configuration"
    else
        print_info ".env file already exists"
    fi
    
    # Create data directories
    print_info "Creating data directories..."
    mkdir -p data/{raw,processed,synthetic,models,outputs}
    mkdir -p logs
    mkdir -p tmp
    
    # Set permissions
    chmod +x scripts/*.sh
    
    print_info "Environment setup completed!"
}

setup_git_hooks() {
    print_header "Setting Up Git Hooks"
    
    # Install pre-commit
    print_info "Installing pre-commit..."
    pre-commit install --install-hooks
    
    # Install commit-msg hook
    print_info "Installing commit-msg hook..."
    pre-commit install --hook-type commit-msg
    
    # Install pre-push hook
    print_info "Installing pre-push hook..."
    pre-commit install --hook-type pre-push
    
    print_info "Git hooks setup completed!"
}

setup_docker() {
    print_header "Setting Up Docker Environment"
    
    # Build development image
    print_info "Building development Docker image..."
    docker build -f Dockerfile.dev -t synthetic-data-guardian:dev .
    
    # Start development services
    print_info "Starting development services..."
    docker-compose -f docker-compose.dev.yml up -d
    
    # Wait for services to be ready
    print_info "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    print_info "Checking service health..."
    docker-compose -f docker-compose.dev.yml ps
    
    print_info "Docker environment setup completed!"
}

run_initial_tests() {
    print_header "Running Initial Tests"
    
    # Run linting
    print_info "Running linters..."
    make lint || print_warning "Linting failed - this is expected for a new project"
    
    # Run tests
    print_info "Running tests..."
    make test || print_warning "Tests failed - this is expected for a new project"
    
    # Check formatting
    print_info "Checking code formatting..."
    make format-check || print_warning "Formatting check failed - this is expected for a new project"
    
    print_info "Initial tests completed!"
}

show_next_steps() {
    print_header "Setup Complete!"
    
    echo -e "${GREEN}🎉 Development environment setup completed successfully!${RESET}"
    echo
    echo -e "${CYAN}Next steps:${RESET}"
    echo "1. Update .env file with your configuration"
    echo "2. Review and customize the configuration files"
    echo "3. Start development: make dev"
    echo "4. Run tests: make test"
    echo "5. Format code: make format"
    echo "6. View logs: make dev-logs"
    echo
    echo -e "${CYAN}Useful commands:${RESET}"
    echo "  make help          - Show all available commands"
    echo "  make dev           - Start development environment"
    echo "  make test          - Run all tests"
    echo "  make lint          - Run linting"
    echo "  make clean         - Clean build artifacts"
    echo
    echo -e "${CYAN}Documentation:${RESET}"
    echo "  README.md          - Project overview"
    echo "  docs/DEVELOPMENT.md - Development guide"
    echo "  CONTRIBUTING.md    - Contribution guidelines"
    echo
    echo -e "${GREEN}Happy coding! 🚀${RESET}"
}

# Main execution
main() {
    print_header "$PROJECT_NAME - Development Environment Setup"
    
    # Check if we're in the right directory
    if [ ! -f "package.json" ] || [ ! -f "pyproject.toml" ]; then
        print_error "This script must be run from the project root directory"
        exit 1
    fi
    
    # Run setup steps
    install_dependencies
    setup_environment
    setup_git_hooks
    
    # Ask about Docker setup
    read -p "Do you want to set up the Docker development environment? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_docker
    else
        print_info "Skipping Docker setup"
    fi
    
    # Ask about running tests
    read -p "Do you want to run initial tests? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_initial_tests
    else
        print_info "Skipping initial tests"
    fi
    
    show_next_steps
}

# Handle script interruption
trap 'print_error "Setup interrupted"; exit 1' INT TERM

# Run main function
main "$@"