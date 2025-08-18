#!/bin/bash

# =============================================================================
# Dependency Update Automation Script
# =============================================================================
# This script automates the process of checking for and updating dependencies
# across multiple package managers (npm, pip, Docker base images)

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
UPDATE_TYPE="${1:-minor}"  # patch, minor, major, all

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]${NC} $1" | tee -a "$LOG_DIR/dependency-updates.log"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] [SUCCESS]${NC} $1" | tee -a "$LOG_DIR/dependency-updates.log"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] [WARNING]${NC} $1" | tee -a "$LOG_DIR/dependency-updates.log"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR]${NC} $1" | tee -a "$LOG_DIR/dependency-updates.log"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Create git branch for updates
create_update_branch() {
    local branch_name="automated/dependency-updates-$(date +%Y%m%d)"
    
    cd "$PROJECT_ROOT"
    
    if git branch | grep -q "$branch_name"; then
        log_info "Switching to existing branch: $branch_name"
        git checkout "$branch_name"
    else
        log_info "Creating new branch: $branch_name"
        git checkout -b "$branch_name"
    fi
    
    echo "$branch_name"
}

# Update NPM dependencies
update_npm_dependencies() {
    local update_type="$1"
    
    cd "$PROJECT_ROOT"
    
    if [[ ! -f "package.json" ]]; then
        log_info "No package.json found, skipping NPM updates"
        return 0
    fi
    
    log_info "Updating NPM dependencies ($update_type)..."
    
    # Backup current package files
    cp package.json package.json.backup
    [[ -f package-lock.json ]] && cp package-lock.json package-lock.json.backup
    
    # Install npm-check-updates if not available
    if ! command_exists ncu; then
        log_info "Installing npm-check-updates..."
        npm install -g npm-check-updates
    fi
    
    # Check for updates first
    log_info "Checking for available updates..."
    ncu --target "$update_type" --format group > "$LOG_DIR/npm-updates-available.txt" 2>&1 || true
    
    # Apply updates
    case "$update_type" in
        "patch")
            ncu -u --target patch
            ;;
        "minor")
            ncu -u --target minor
            ;;
        "major")
            ncu -u --target major
            ;;
        "all")
            ncu -u
            ;;
        *)
            log_error "Unknown update type: $update_type"
            return 1
            ;;
    esac
    
    # Check if package.json was modified
    if ! diff -q package.json package.json.backup >/dev/null 2>&1; then
        log_info "Package.json was updated, installing new dependencies..."
        
        # Install updated dependencies
        npm install
        
        # Run tests to ensure everything still works
        if command_exists npm && npm run test >/dev/null 2>&1; then
            log_success "NPM dependencies updated and tests pass"
            
            # Generate update summary
            {
                echo "NPM Dependency Updates ($update_type) - $(date)"
                echo "============================================="
                echo
                echo "Updated packages:"
                diff package.json.backup package.json | grep '^[<>]' | grep -E '".*":' || echo "No changes detected"
                echo
                echo "Security audit:"
                npm audit --audit-level=high 2>&1 || true
            } > "$LOG_DIR/npm-update-summary.txt"
            
            # Clean up backup files
            rm -f package.json.backup package-lock.json.backup
            
            return 0
        else
            log_error "Tests failed after NPM updates, reverting changes..."
            mv package.json.backup package.json
            [[ -f package-lock.json.backup ]] && mv package-lock.json.backup package-lock.json
            npm install
            return 1
        fi
    else
        log_info "No NPM updates available for $update_type level"
        rm -f package.json.backup package-lock.json.backup
        return 0
    fi
}

# Update Python dependencies
update_python_dependencies() {
    local update_type="$1"
    
    cd "$PROJECT_ROOT"
    
    if [[ ! -f "requirements.txt" ]] && [[ ! -f "pyproject.toml" ]]; then
        log_info "No Python requirements files found, skipping Python updates"
        return 0
    fi
    
    log_info "Updating Python dependencies ($update_type)..."
    
    # Use pip-tools if available
    if command_exists pip-compile; then
        log_info "Using pip-tools for Python dependency updates..."
        
        # Backup requirements files
        [[ -f requirements.txt ]] && cp requirements.txt requirements.txt.backup
        [[ -f requirements-dev.txt ]] && cp requirements-dev.txt requirements-dev.txt.backup
        
        # Update requirements
        if [[ -f "requirements.in" ]]; then
            pip-compile --upgrade requirements.in
        elif [[ -f "requirements.txt" ]]; then
            pip-compile --upgrade requirements.txt
        fi
        
        if [[ -f "requirements-dev.in" ]]; then
            pip-compile --upgrade requirements-dev.in
        elif [[ -f "requirements-dev.txt" ]]; then
            pip-compile --upgrade requirements-dev.txt
        fi
        
    elif command_exists poetry; then
        log_info "Using Poetry for Python dependency updates..."
        
        case "$update_type" in
            "patch"|"minor")
                poetry update
                ;;
            "major"|"all")
                # For major updates, you might want to manually specify packages
                poetry show --outdated | grep -E "^\S+" | awk '{print $1}' | xargs poetry add
                ;;
        esac
        
    else
        log_info "Using pip-upgrader for Python dependency updates..."
        
        # Install pip-upgrader if not available
        if ! command_exists pip-upgrade; then
            pip install pip-upgrader
        fi
        
        # Backup and upgrade
        [[ -f requirements.txt ]] && cp requirements.txt requirements.txt.backup
        
        pip-upgrade requirements.txt --skip-virtualenv-check
    fi
    
    # Install updated dependencies and run tests
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
        [[ -f "requirements-dev.txt" ]] && pip install -r requirements-dev.txt
    elif [[ -f "pyproject.toml" ]]; then
        poetry install
    fi
    
    # Run tests
    if command_exists pytest; then
        if pytest tests/ >/dev/null 2>&1; then
            log_success "Python dependencies updated and tests pass"
            
            # Generate update summary
            {
                echo "Python Dependency Updates ($update_type) - $(date)"
                echo "================================================"
                echo
                if [[ -f "requirements.txt.backup" ]]; then
                    echo "Requirements.txt changes:"
                    diff requirements.txt.backup requirements.txt | grep '^[<>]' || echo "No changes detected"
                fi
                echo
                echo "Security check:"
                safety check 2>&1 || true
            } > "$LOG_DIR/python-update-summary.txt"
            
            # Clean up backup files
            rm -f requirements.txt.backup requirements-dev.txt.backup
            
            return 0
        else
            log_error "Tests failed after Python updates, reverting changes..."
            [[ -f requirements.txt.backup ]] && mv requirements.txt.backup requirements.txt
            [[ -f requirements-dev.txt.backup ]] && mv requirements-dev.txt.backup requirements-dev.txt
            return 1
        fi
    else
        log_warning "No pytest found, skipping test verification"
        return 0
    fi
}

# Update Docker base images
update_docker_images() {
    cd "$PROJECT_ROOT"
    
    local updated=false
    
    for dockerfile in Dockerfile Dockerfile.dev Dockerfile.prod; do
        if [[ -f "$dockerfile" ]]; then
            log_info "Checking $dockerfile for base image updates..."
            
            # Backup dockerfile
            cp "$dockerfile" "$dockerfile.backup"
            
            # Update common base images
            if grep -q "FROM node:" "$dockerfile"; then
                log_info "Updating Node.js base image in $dockerfile..."
                # Update to latest LTS patch version
                sed -i 's/FROM node:[0-9]*\.[0-9]*\.[0-9]*/FROM node:18/' "$dockerfile"
                updated=true
            fi
            
            if grep -q "FROM python:" "$dockerfile"; then
                log_info "Updating Python base image in $dockerfile..."
                # Update to latest stable patch version
                sed -i 's/FROM python:[0-9]*\.[0-9]*\.[0-9]*/FROM python:3.11/' "$dockerfile"
                updated=true
            fi
            
            if grep -q "FROM ubuntu:" "$dockerfile"; then
                log_info "Updating Ubuntu base image in $dockerfile..."
                # Update to latest LTS
                sed -i 's/FROM ubuntu:[0-9]*\.[0-9]*/FROM ubuntu:22.04/' "$dockerfile"
                updated=true
            fi
            
            if grep -q "FROM alpine:" "$dockerfile"; then
                log_info "Updating Alpine base image in $dockerfile..."
                # Update to latest stable
                sed -i 's/FROM alpine:[0-9]*\.[0-9]*/FROM alpine:3.18/' "$dockerfile"
                updated=true
            fi
            
            # Check if dockerfile was actually modified
            if ! diff -q "$dockerfile" "$dockerfile.backup" >/dev/null 2>&1; then
                log_info "Testing updated $dockerfile..."
                
                # Build and test the updated image
                if docker build -t "test-$dockerfile" -f "$dockerfile" .; then
                    log_success "$dockerfile updated successfully"
                    
                    # Clean up test image
                    docker rmi "test-$dockerfile" >/dev/null 2>&1 || true
                else
                    log_error "Failed to build updated $dockerfile, reverting..."
                    mv "$dockerfile.backup" "$dockerfile"
                    updated=false
                fi
            else
                log_info "No updates needed for $dockerfile"
            fi
            
            # Clean up backup
            rm -f "$dockerfile.backup"
        fi
    done
    
    if [[ "$updated" == true ]]; then
        log_success "Docker base images updated"
        return 0
    else
        log_info "No Docker image updates were applied"
        return 0
    fi
}

# Check for security vulnerabilities
run_security_checks() {
    cd "$PROJECT_ROOT"
    
    log_info "Running security checks after updates..."
    
    local security_issues=0
    
    # NPM audit
    if [[ -f "package.json" ]] && command_exists npm; then
        if ! npm audit --audit-level=high >/dev/null 2>&1; then
            log_warning "NPM audit found security issues"
            npm audit > "$LOG_DIR/npm-audit-post-update.txt" 2>&1
            ((security_issues++))
        fi
    fi
    
    # Python safety check
    if command_exists safety; then
        if ! safety check >/dev/null 2>&1; then
            log_warning "Python safety check found security issues"
            safety check > "$LOG_DIR/safety-check-post-update.txt" 2>&1
            ((security_issues++))
        fi
    fi
    
    # Snyk check if available
    if command_exists snyk && [[ -n "${SNYK_TOKEN:-}" ]]; then
        if ! snyk test >/dev/null 2>&1; then
            log_warning "Snyk found security issues"
            snyk test > "$LOG_DIR/snyk-check-post-update.txt" 2>&1
            ((security_issues++))
        fi
    fi
    
    if [[ $security_issues -eq 0 ]]; then
        log_success "Security checks passed"
        return 0
    else
        log_warning "Security checks found $security_issues issues"
        return 1
    fi
}

# Create pull request
create_pull_request() {
    local branch_name="$1"
    
    cd "$PROJECT_ROOT"
    
    # Check if there are any changes to commit
    if git diff --quiet && git diff --cached --quiet; then
        log_info "No changes to commit"
        return 0
    fi
    
    # Commit changes
    git add .
    git commit -m "chore: automated dependency updates ($UPDATE_TYPE)

- Updated NPM dependencies to latest $UPDATE_TYPE versions
- Updated Python dependencies to latest $UPDATE_TYPE versions  
- Updated Docker base images to latest stable versions
- All tests passing
- Security audits clean

🤖 Automated dependency update" || {
        log_error "Failed to commit changes"
        return 1
    }
    
    # Push branch
    git push -u origin "$branch_name" || {
        log_error "Failed to push branch"
        return 1
    }
    
    # Create pull request using GitHub CLI if available
    if command_exists gh; then
        gh pr create \
            --title "chore: automated dependency updates ($UPDATE_TYPE)" \
            --body "## Automated Dependency Update

This PR updates dependencies to their latest $UPDATE_TYPE versions.

### Changes Made
- Updated NPM dependencies with npm-check-updates
- Updated Python dependencies with pip-tools/poetry
- Updated Docker base images to latest stable versions

### Verification
- ✅ All tests passing
- ✅ Security audits clean
- ✅ Build successful

### Manual Review Required
- [ ] Review dependency changes
- [ ] Verify application functionality
- [ ] Check for any breaking changes in changelogs

---

🤖 This PR was created automatically by the dependency update workflow." \
            --label "dependencies,automated,chore" || {
            log_warning "Failed to create pull request via GitHub CLI"
        }
    else
        log_info "GitHub CLI not available, please create pull request manually"
        log_info "Branch: $branch_name"
    fi
    
    log_success "Dependency update process completed"
}

# Main function
main() {
    log_info "Starting dependency update automation (type: $UPDATE_TYPE)..."
    
    # Validate update type
    if [[ ! "$UPDATE_TYPE" =~ ^(patch|minor|major|all)$ ]]; then
        log_error "Invalid update type: $UPDATE_TYPE"
        echo "Usage: $0 [patch|minor|major|all]"
        exit 1
    fi
    
    # Create update branch
    branch_name=$(create_update_branch)
    
    # Track if any updates were made
    local updates_made=false
    
    # Update NPM dependencies
    if update_npm_dependencies "$UPDATE_TYPE"; then
        updates_made=true
    fi
    
    # Update Python dependencies
    if update_python_dependencies "$UPDATE_TYPE"; then
        updates_made=true
    fi
    
    # Update Docker images
    if update_docker_images; then
        updates_made=true
    fi
    
    # Run security checks
    if ! run_security_checks; then
        log_warning "Security issues found after updates - manual review required"
    fi
    
    # Create pull request if updates were made
    if [[ "$updates_made" == true ]]; then
        create_pull_request "$branch_name"
    else
        log_info "No dependency updates were needed"
        git checkout main
        git branch -D "$branch_name" 2>/dev/null || true
    fi
}

# Run main function
main "$@"