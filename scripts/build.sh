#!/bin/bash
# =============================================================================
# Synthetic Data Guardian - Build Script
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
VERSION="${VERSION:-$(git describe --tags --always --dirty)}"
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT=$(git rev-parse --short HEAD)
BUILD_ARGS=""

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
    
    command -v docker >/dev/null 2>&1 || { log_error "Docker is required but not installed. Aborting."; exit 1; }
    command -v git >/dev/null 2>&1 || { log_error "Git is required but not installed. Aborting."; exit 1; }
    
    log_success "Dependencies check passed"
}

validate_git_state() {
    log_step "Validating git state..."
    
    if [[ -n $(git status --porcelain) ]]; then
        log_warning "Working directory is not clean. This may affect build reproducibility."
        
        # Ask user if they want to continue
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_error "Build cancelled by user"
            exit 1
        fi
    fi
    
    log_success "Git state validation passed"
}

build_image() {
    local target=$1
    local tag_suffix=$2
    local dockerfile=${3:-Dockerfile}
    
    log_step "Building ${target} image..."
    
    local image_tag="${REGISTRY}/${APP_NAME}:${VERSION}${tag_suffix}"
    local latest_tag="${REGISTRY}/${APP_NAME}:latest${tag_suffix}"
    
    # Build the image
    docker build \
        --target="${target}" \
        --dockerfile="${dockerfile}" \
        --tag="${image_tag}" \
        --tag="${latest_tag}" \
        --build-arg="VERSION=${VERSION}" \
        --build-arg="BUILD_DATE=${BUILD_DATE}" \
        --build-arg="GIT_COMMIT=${GIT_COMMIT}" \
        --build-arg="APP_NAME=${APP_NAME}" \
        ${BUILD_ARGS} \
        .
    
    log_success "Built ${target} image: ${image_tag}"
}

scan_image() {
    local image_tag=$1
    
    log_step "Scanning image for vulnerabilities..."
    
    # Use trivy if available, otherwise skip
    if command -v trivy >/dev/null 2>&1; then
        trivy image --severity HIGH,CRITICAL "${image_tag}" || {
            log_warning "Security scan found issues. Review the output above."
        }
    else
        log_warning "Trivy not available. Skipping security scan."
        log_info "Install trivy for security scanning: https://aquasecurity.github.io/trivy/"
    fi
}

push_image() {
    local image_tag=$1
    
    log_step "Pushing image to registry..."
    
    # Check if user is logged in to registry
    if ! docker info | grep -q "Username"; then
        log_warning "Not logged in to Docker registry. You may need to login:"
        log_info "  docker login ${REGISTRY}"
        
        read -p "Continue with push? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping push"
            return 0
        fi
    fi
    
    docker push "${image_tag}"
    docker push "${REGISTRY}/${APP_NAME}:latest"
    
    log_success "Pushed image: ${image_tag}"
}

generate_sbom() {
    local image_tag=$1
    
    log_step "Generating Software Bill of Materials (SBOM)..."
    
    # Use syft if available
    if command -v syft >/dev/null 2>&1; then
        syft "${image_tag}" -o spdx-json > "sbom-${VERSION}.json"
        log_success "SBOM generated: sbom-${VERSION}.json"
    else
        log_warning "Syft not available. Skipping SBOM generation."
        log_info "Install syft for SBOM generation: https://github.com/anchore/syft"
    fi
}

cleanup() {
    log_step "Cleaning up..."
    
    # Remove dangling images
    docker image prune -f >/dev/null 2>&1 || true
    
    log_success "Cleanup completed"
}

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] [TARGET]

Build script for Synthetic Data Guardian

OPTIONS:
    -h, --help          Show this help message
    -v, --version       Set version tag (default: git describe)
    -r, --registry      Set Docker registry (default: ghcr.io/danieleschmidt)
    -p, --push          Push image to registry after build
    -s, --scan          Scan image for vulnerabilities
    --sbom              Generate Software Bill of Materials
    --no-cache          Build without cache
    --dev               Build development image
    --multi-arch        Build multi-architecture images (experimental)

TARGETS:
    production          Build production image (default)
    development         Build development image
    all                 Build all images

EXAMPLES:
    $0                          # Build production image
    $0 --push                   # Build and push production image
    $0 --dev --scan             # Build development image with security scan
    $0 -v v1.2.3 --push        # Build with specific version and push
    $0 all --push --sbom        # Build all images, push, and generate SBOM

EOF
}

main() {
    local target="production"
    local push_image_flag=false
    local scan_image_flag=false
    local generate_sbom_flag=false
    local no_cache_flag=false
    local dev_flag=false
    local multi_arch_flag=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -p|--push)
                push_image_flag=true
                shift
                ;;
            -s|--scan)
                scan_image_flag=true
                shift
                ;;
            --sbom)
                generate_sbom_flag=true
                shift
                ;;
            --no-cache)
                BUILD_ARGS="${BUILD_ARGS} --no-cache"
                shift
                ;;
            --dev)
                dev_flag=true
                target="development"
                shift
                ;;
            --multi-arch)
                multi_arch_flag=true
                BUILD_ARGS="${BUILD_ARGS} --platform=linux/amd64,linux/arm64"
                shift
                ;;
            production|development|all)
                target="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Print build information
    log_info "Build Configuration:"
    log_info "  App Name: ${APP_NAME}"
    log_info "  Version: ${VERSION}"
    log_info "  Registry: ${REGISTRY}"
    log_info "  Target: ${target}"
    log_info "  Build Date: ${BUILD_DATE}"
    log_info "  Git Commit: ${GIT_COMMIT}"
    echo
    
    # Pre-build checks
    check_dependencies
    validate_git_state
    
    # Build images
    case "${target}" in
        "production")
            build_image "production" ""
            if [[ "${scan_image_flag}" == true ]]; then
                scan_image "${REGISTRY}/${APP_NAME}:${VERSION}"
            fi
            if [[ "${generate_sbom_flag}" == true ]]; then
                generate_sbom "${REGISTRY}/${APP_NAME}:${VERSION}"
            fi
            if [[ "${push_image_flag}" == true ]]; then
                push_image "${REGISTRY}/${APP_NAME}:${VERSION}"
            fi
            ;;
        "development")
            build_image "development" "-dev" "Dockerfile.dev"
            if [[ "${scan_image_flag}" == true ]]; then
                scan_image "${REGISTRY}/${APP_NAME}:${VERSION}-dev"
            fi
            if [[ "${push_image_flag}" == true ]]; then
                push_image "${REGISTRY}/${APP_NAME}:${VERSION}-dev"
            fi
            ;;
        "all")
            build_image "production" ""
            build_image "development" "-dev" "Dockerfile.dev"
            
            if [[ "${scan_image_flag}" == true ]]; then
                scan_image "${REGISTRY}/${APP_NAME}:${VERSION}"
                scan_image "${REGISTRY}/${APP_NAME}:${VERSION}-dev"
            fi
            
            if [[ "${generate_sbom_flag}" == true ]]; then
                generate_sbom "${REGISTRY}/${APP_NAME}:${VERSION}"
            fi
            
            if [[ "${push_image_flag}" == true ]]; then
                push_image "${REGISTRY}/${APP_NAME}:${VERSION}"
                push_image "${REGISTRY}/${APP_NAME}:${VERSION}-dev"
            fi
            ;;
        *)
            log_error "Unknown target: ${target}"
            exit 1
            ;;
    esac
    
    # Cleanup
    cleanup
    
    log_success "Build completed successfully!"
    
    # Show next steps
    echo
    log_info "Next steps:"
    if [[ "${push_image_flag}" == false ]]; then
        log_info "  - Push to registry: docker push ${REGISTRY}/${APP_NAME}:${VERSION}"
    fi
    log_info "  - Run locally: docker run -p 8080:8080 ${REGISTRY}/${APP_NAME}:${VERSION}"
    log_info "  - Deploy to production: kubectl apply -f k8s/"
}

# Run main function
main "$@"