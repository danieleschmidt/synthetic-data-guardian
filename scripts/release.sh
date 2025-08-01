#!/bin/bash
# =============================================================================
# Synthetic Data Guardian - Release Script
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="synthetic-data-guardian"
REGISTRY="${DOCKER_REGISTRY:-ghcr.io/danieleschmidt}"
GITHUB_REPO="${GITHUB_REPOSITORY:-danieleschmidt/synthetic-data-guardian}"

# Functions
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

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

check_dependencies() {
    log_step "Checking dependencies..."
    
    command -v git >/dev/null 2>&1 || { log_error "Git is required but not installed. Aborting."; exit 1; }
    command -v jq >/dev/null 2>&1 || { log_error "jq is required but not installed. Aborting."; exit 1; }
    command -v gh >/dev/null 2>&1 || { log_error "GitHub CLI is required but not installed. Aborting."; exit 1; }
    
    log_success "Dependencies check passed"
}

validate_git_state() {
    log_step "Validating git state..."
    
    # Check if on main branch
    current_branch=$(git branch --show-current)
    if [[ "${current_branch}" != "main" ]]; then
        log_error "Must be on main branch to create release. Current branch: ${current_branch}"
        exit 1
    fi
    
    # Check if working directory is clean
    if [[ -n $(git status --porcelain) ]]; then
        log_error "Working directory is not clean. Please commit or stash changes."
        exit 1
    fi
    
    # Check if up to date with remote
    git fetch origin main
    if [[ $(git rev-parse HEAD) != $(git rev-parse origin/main) ]]; then
        log_error "Local main branch is not up to date with origin/main"
        exit 1
    fi
    
    log_success "Git state validation passed"
}

get_current_version() {
    # Get version from package.json
    if [[ -f "package.json" ]]; then
        jq -r '.version' package.json
    else
        log_error "package.json not found"
        exit 1
    fi
}

validate_version() {
    local version=$1
    
    # Check if version follows semantic versioning
    if [[ ! $version =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$ ]]; then
        log_error "Version must follow semantic versioning (e.g., 1.2.3 or 1.2.3-beta)"
        exit 1
    fi
    
    # Check if version tag already exists
    if git tag | grep -q "^v${version}$"; then
        log_error "Version ${version} already exists"
        exit 1
    fi
    
    # Check if version is greater than current
    current_version=$(get_current_version)
    if [[ "${version}" == "${current_version}" ]]; then
        log_error "Version ${version} is the same as current version"
        exit 1
    fi
}

update_version() {
    local version=$1
    
    log_step "Updating version to ${version}..."
    
    # Update package.json
    if [[ -f "package.json" ]]; then
        jq ".version = \"${version}\"" package.json > package.json.tmp
        mv package.json.tmp package.json
    fi
    
    # Update pyproject.toml if it exists
    if [[ -f "pyproject.toml" ]]; then
        sed -i "s/^version = .*/version = \"${version}\"/" pyproject.toml
    fi
    
    # Update version in other files if needed
    if [[ -f "src/version.py" ]]; then
        echo "__version__ = \"${version}\"" > src/version.py
    fi
    
    log_success "Version updated to ${version}"
}

generate_changelog() {
    local version=$1
    local previous_tag=$2
    
    log_step "Generating changelog..."
    
    local changelog_file="RELEASE_NOTES_${version}.md"
    
    cat > "${changelog_file}" << EOF
# Release Notes for ${version}

## What's Changed

EOF
    
    # Get commits since last tag
    if [[ -n "${previous_tag}" ]]; then
        git log --pretty=format:"* %s" "${previous_tag}..HEAD" >> "${changelog_file}"
    else
        git log --pretty=format:"* %s" >> "${changelog_file}"
    fi
    
    cat >> "${changelog_file}" << EOF

## Docker Images

\`\`\`
docker pull ${REGISTRY}/${APP_NAME}:${version}
docker pull ${REGISTRY}/${APP_NAME}:latest
\`\`\`

## Installation

### Using Docker
\`\`\`bash
docker run -p 8080:8080 ${REGISTRY}/${APP_NAME}:${version}
\`\`\`

### Using Python Package
\`\`\`bash
pip install ${APP_NAME}==${version}
\`\`\`

---

**Full Changelog**: https://github.com/${GITHUB_REPO}/compare/${previous_tag}...v${version}
EOF
    
    log_success "Changelog generated: ${changelog_file}"
    echo "${changelog_file}"
}

run_tests() {
    log_step "Running tests..."
    
    # Run tests using make if available
    if [[ -f "Makefile" ]]; then
        make test || {
            log_error "Tests failed. Aborting release."
            exit 1
        }
    else
        # Fallback to npm/python tests
        if [[ -f "package.json" ]]; then
            npm test || {
                log_error "JavaScript tests failed. Aborting release."
                exit 1
            }
        fi
        
        if [[ -f "pytest.ini" ]] || [[ -f "pyproject.toml" ]]; then
            python -m pytest || {
                log_error "Python tests failed. Aborting release."
                exit 1
            }
        fi
    fi
    
    log_success "All tests passed"
}

build_and_push() {
    local version=$1
    
    log_step "Building and pushing Docker images..."
    
    # Use build script if available
    if [[ -f "scripts/build.sh" ]]; then
        ./scripts/build.sh --version="${version}" --push --scan --sbom
    else
        # Fallback build
        docker build -t "${REGISTRY}/${APP_NAME}:${version}" .
        docker tag "${REGISTRY}/${APP_NAME}:${version}" "${REGISTRY}/${APP_NAME}:latest"
        docker push "${REGISTRY}/${APP_NAME}:${version}"
        docker push "${REGISTRY}/${APP_NAME}:latest"
    fi
    
    log_success "Docker images built and pushed"
}

create_git_tag() {
    local version=$1
    local changelog_file=$2
    
    log_step "Creating git tag..."
    
    # Commit version changes
    git add -A
    git commit -m "chore: bump version to ${version}" || true
    
    # Create annotated tag
    git tag -a "v${version}" -F "${changelog_file}"
    
    # Push tag
    git push origin "v${version}"
    git push origin main
    
    log_success "Git tag v${version} created and pushed"
}

create_github_release() {
    local version=$1
    local changelog_file=$2
    
    log_step "Creating GitHub release..."
    
    # Create GitHub release
    gh release create "v${version}" \
        --title "Release ${version}" \
        --notes-file "${changelog_file}" \
        --latest
    
    # Upload additional assets if they exist
    if [[ -f "sbom-${version}.json" ]]; then
        gh release upload "v${version}" "sbom-${version}.json"
    fi
    
    log_success "GitHub release created: https://github.com/${GITHUB_REPO}/releases/tag/v${version}"
}

cleanup() {
    log_step "Cleaning up temporary files..."
    
    # Remove temporary changelog files
    rm -f RELEASE_NOTES_*.md
    
    log_success "Cleanup completed"
}

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] VERSION

Create a new release of Synthetic Data Guardian

OPTIONS:
    -h, --help          Show this help message
    --dry-run           Show what would be done without making changes
    --skip-tests        Skip running tests
    --skip-build        Skip building Docker images
    --force             Force release even if validation fails

EXAMPLES:
    $0 1.2.3                    # Create release 1.2.3
    $0 --dry-run 1.2.3          # Preview release 1.2.3
    $0 --skip-tests 1.2.3-beta  # Create beta release without tests

EOF
}

main() {
    local version=""
    local dry_run=false
    local skip_tests=false
    local skip_build=false
    local force=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --skip-tests)
                skip_tests=true
                shift
                ;;
            --skip-build)
                skip_build=true
                shift
                ;;
            --force)
                force=true
                shift
                ;;
            *)
                if [[ -z "${version}" ]]; then
                    version="$1"
                else
                    log_error "Unknown option: $1"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Check if version is provided
    if [[ -z "${version}" ]]; then
        log_error "Version is required"
        show_usage
        exit 1
    fi
    
    # Print release information
    log_info "Release Configuration:"
    log_info "  App Name: ${APP_NAME}"
    log_info "  Version: ${version}"
    log_info "  Registry: ${REGISTRY}"
    log_info "  Repository: ${GITHUB_REPO}"
    log_info "  Dry Run: ${dry_run}"
    echo
    
    if [[ "${dry_run}" == true ]]; then
        log_warning "DRY RUN MODE - No changes will be made"
        echo
    fi
    
    # Pre-release checks
    check_dependencies
    
    if [[ "${force}" == false ]]; then
        validate_git_state
        validate_version "${version}"
    fi
    
    # Get previous tag for changelog
    previous_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
    
    if [[ "${dry_run}" == true ]]; then
        log_info "Would perform the following actions:"
        log_info "  1. Update version in package files"
        log_info "  2. Run tests (skip: ${skip_tests})"
        log_info "  3. Build and push Docker images (skip: ${skip_build})"
        log_info "  4. Create git tag v${version}"
        log_info "  5. Create GitHub release"
        log_info "  6. Generate changelog from ${previous_tag:-'beginning'} to HEAD"
        return 0
    fi
    
    # Perform release steps
    update_version "${version}"
    
    if [[ "${skip_tests}" == false ]]; then
        run_tests
    else
        log_warning "Skipping tests"
    fi
    
    changelog_file=$(generate_changelog "${version}" "${previous_tag}")
    
    if [[ "${skip_build}" == false ]]; then
        build_and_push "${version}"
    else
        log_warning "Skipping Docker build"
    fi
    
    create_git_tag "${version}" "${changelog_file}"
    create_github_release "${version}" "${changelog_file}"
    
    # Cleanup
    cleanup
    
    log_success "Release ${version} completed successfully!"
    
    # Show next steps
    echo
    log_info "Release Summary:"
    log_info "  - Version: ${version}"
    log_info "  - Docker: ${REGISTRY}/${APP_NAME}:${version}"
    log_info "  - GitHub: https://github.com/${GITHUB_REPO}/releases/tag/v${version}"
    log_info "  - Changes: ${previous_tag:-'beginning'}..v${version}"
}

# Run main function
main "$@"