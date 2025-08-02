#!/bin/bash

# =============================================================================
# Docker Security Scanning Script
# Comprehensive security scanning for Docker images
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-synthetic-data-guardian}"
TAG="${TAG:-latest}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

# Scan tools configuration
TRIVY_ENABLED="${TRIVY_ENABLED:-true}"
DOCKER_SCOUT_ENABLED="${DOCKER_SCOUT_ENABLED:-true}"
HADOLINT_ENABLED="${HADOLINT_ENABLED:-true}"
SNYK_ENABLED="${SNYK_ENABLED:-false}"  # Requires API key
GRYPE_ENABLED="${GRYPE_ENABLED:-false}"  # Optional

# Severity thresholds
MAX_CRITICAL="${MAX_CRITICAL:-0}"
MAX_HIGH="${MAX_HIGH:-5}"
MAX_MEDIUM="${MAX_MEDIUM:-20}"

# Output directory
OUTPUT_DIR="${PROJECT_ROOT}/security-reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_DIR="${OUTPUT_DIR}/${TIMESTAMP}"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [ "${DEBUG:-false}" = "true" ]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Help function
show_help() {
    cat << EOF
Docker Security Scanner for Synthetic Data Guardian

Usage: $0 [OPTIONS]

Options:
    -i, --image IMAGE       Docker image name (default: synthetic-data-guardian)
    -t, --tag TAG          Image tag (default: latest)
    -o, --output DIR       Output directory for reports (default: ./security-reports)
    --max-critical NUM     Maximum critical vulnerabilities allowed (default: 0)
    --max-high NUM         Maximum high vulnerabilities allowed (default: 5)
    --max-medium NUM       Maximum medium vulnerabilities allowed (default: 20)
    --skip-trivy           Skip Trivy scanning
    --skip-scout           Skip Docker Scout scanning
    --skip-hadolint        Skip Hadolint scanning
    --enable-snyk          Enable Snyk scanning (requires SNYK_TOKEN)
    --enable-grype         Enable Grype scanning
    -h, --help             Show this help message

Environment Variables:
    SNYK_TOKEN            Snyk API token (required for Snyk scanning)
    DOCKER_SCOUT_TOKEN    Docker Scout token (optional)
    TRIVY_DB_REPOSITORY   Custom Trivy DB repository
    DEBUG                 Enable debug output (true/false)

Examples:
    $0                                    # Scan default image with default settings
    $0 -i myapp -t v1.2.3                # Scan specific image and tag
    $0 --enable-snyk --max-critical 0    # Enable Snyk and allow no critical vulns
    $0 --skip-trivy --enable-grype       # Skip Trivy, use Grype instead

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--image)
                IMAGE_NAME="$2"
                FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"
                shift 2
                ;;
            -t|--tag)
                TAG="$2"
                FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                REPORT_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
                shift 2
                ;;
            --max-critical)
                MAX_CRITICAL="$2"
                shift 2
                ;;
            --max-high)
                MAX_HIGH="$2"
                shift 2
                ;;
            --max-medium)
                MAX_MEDIUM="$2"
                shift 2
                ;;
            --skip-trivy)
                TRIVY_ENABLED=false
                shift
                ;;
            --skip-scout)
                DOCKER_SCOUT_ENABLED=false
                shift
                ;;
            --skip-hadolint)
                HADOLINT_ENABLED=false
                shift
                ;;
            --enable-snyk)
                SNYK_ENABLED=true
                shift
                ;;
            --enable-grype)
                GRYPE_ENABLED=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Check tool availability
check_tools() {
    log_info "Checking security tool availability..."
    
    local missing_tools=()
    
    # Required tools
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi
    
    # Optional tools
    if [ "$TRIVY_ENABLED" = "true" ] && ! command -v trivy &> /dev/null; then
        log_warn "Trivy not found. Installing..."
        install_trivy
    fi
    
    if [ "$DOCKER_SCOUT_ENABLED" = "true" ] && ! docker scout version &> /dev/null; then
        log_warn "Docker Scout not available. Skipping Docker Scout scan."
        DOCKER_SCOUT_ENABLED=false
    fi
    
    if [ "$HADOLINT_ENABLED" = "true" ] && ! command -v hadolint &> /dev/null; then
        log_warn "Hadolint not found. Installing..."
        install_hadolint
    fi
    
    if [ "$SNYK_ENABLED" = "true" ]; then
        if ! command -v snyk &> /dev/null; then
            log_warn "Snyk not found. Installing..."
            install_snyk
        fi
        
        if [ -z "$SNYK_TOKEN" ]; then
            log_error "SNYK_TOKEN environment variable required for Snyk scanning"
            SNYK_ENABLED=false
        fi
    fi
    
    if [ "$GRYPE_ENABLED" = "true" ] && ! command -v grype &> /dev/null; then
        log_warn "Grype not found. Installing..."
        install_grype
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    log_info "Tool availability check completed"
}

# Install Trivy
install_trivy() {
    log_info "Installing Trivy..."
    
    case "$(uname -s)" in
        Linux*)
            curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
            ;;
        Darwin*)
            if command -v brew &> /dev/null; then
                brew install trivy
            else
                log_error "Homebrew not found. Please install Trivy manually."
                exit 1
            fi
            ;;
        *)
            log_error "Unsupported OS for automatic Trivy installation"
            exit 1
            ;;
    esac
}

# Install Hadolint
install_hadolint() {
    log_info "Installing Hadolint..."
    
    case "$(uname -s)" in
        Linux*)
            wget -O /usr/local/bin/hadolint https://github.com/hadolint/hadolint/releases/latest/download/hadolint-Linux-x86_64
            chmod +x /usr/local/bin/hadolint
            ;;
        Darwin*)
            if command -v brew &> /dev/null; then
                brew install hadolint
            else
                log_error "Homebrew not found. Please install Hadolint manually."
                exit 1
            fi
            ;;
        *)
            log_error "Unsupported OS for automatic Hadolint installation"
            exit 1
            ;;
    esac
}

# Install Snyk
install_snyk() {
    log_info "Installing Snyk..."
    npm install -g snyk
}

# Install Grype
install_grype() {
    log_info "Installing Grype..."
    curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin
}

# Setup output directory
setup_output() {
    log_info "Setting up output directory: $REPORT_DIR"
    mkdir -p "$REPORT_DIR"
    
    # Create subdirectories for different scan types
    mkdir -p "$REPORT_DIR/trivy"
    mkdir -p "$REPORT_DIR/scout"
    mkdir -p "$REPORT_DIR/hadolint"
    mkdir -p "$REPORT_DIR/snyk"
    mkdir -p "$REPORT_DIR/grype"
}

# Check if image exists
check_image() {
    log_info "Checking if image exists: $FULL_IMAGE_NAME"
    
    if ! docker image inspect "$FULL_IMAGE_NAME" &> /dev/null; then
        log_error "Image not found: $FULL_IMAGE_NAME"
        log_info "Available images:"
        docker images
        exit 1
    fi
    
    log_info "Image found: $FULL_IMAGE_NAME"
}

# Dockerfile linting with Hadolint
scan_dockerfile() {
    if [ "$HADOLINT_ENABLED" != "true" ]; then
        log_info "Skipping Dockerfile scan (Hadolint disabled)"
        return 0
    fi
    
    log_info "Scanning Dockerfile with Hadolint..."
    
    local dockerfile_path="${PROJECT_ROOT}/Dockerfile"
    local output_file="${REPORT_DIR}/hadolint/hadolint-report.json"
    
    if [ ! -f "$dockerfile_path" ]; then
        log_warn "Dockerfile not found at $dockerfile_path"
        return 0
    fi
    
    if hadolint --format json "$dockerfile_path" > "$output_file" 2>&1; then
        log_info "Hadolint scan completed successfully"
        log_info "Report saved to: $output_file"
    else
        log_warn "Hadolint found issues in Dockerfile"
        log_info "Report saved to: $output_file"
        
        # Show summary
        if [ -s "$output_file" ]; then
            local issue_count=$(jq length "$output_file" 2>/dev/null || echo "unknown")
            log_warn "Found $issue_count Dockerfile issues"
        fi
    fi
}

# Vulnerability scanning with Trivy
scan_with_trivy() {
    if [ "$TRIVY_ENABLED" != "true" ]; then
        log_info "Skipping Trivy scan (disabled)"
        return 0
    fi
    
    log_info "Scanning with Trivy..."
    
    local json_output="${REPORT_DIR}/trivy/trivy-report.json"
    local table_output="${REPORT_DIR}/trivy/trivy-report.txt"
    local sarif_output="${REPORT_DIR}/trivy/trivy-report.sarif"
    
    # Update Trivy database
    log_info "Updating Trivy database..."
    trivy image --download-db-only
    
    # Scan for vulnerabilities
    log_info "Running Trivy vulnerability scan..."
    trivy image --format json --output "$json_output" "$FULL_IMAGE_NAME"
    trivy image --format table --output "$table_output" "$FULL_IMAGE_NAME"
    trivy image --format sarif --output "$sarif_output" "$FULL_IMAGE_NAME"
    
    # Scan for secrets
    log_info "Running Trivy secrets scan..."
    trivy image --scanners secret --format json --output "${REPORT_DIR}/trivy/trivy-secrets.json" "$FULL_IMAGE_NAME"
    
    # Scan for configuration issues
    log_info "Running Trivy config scan..."
    trivy image --scanners config --format json --output "${REPORT_DIR}/trivy/trivy-config.json" "$FULL_IMAGE_NAME"
    
    log_info "Trivy scan completed"
    log_info "Reports saved to: ${REPORT_DIR}/trivy/"
    
    # Parse results for summary
    if [ -f "$json_output" ]; then
        parse_trivy_results "$json_output"
    fi
}

# Parse Trivy results
parse_trivy_results() {
    local json_file="$1"
    
    if ! command -v jq &> /dev/null; then
        log_warn "jq not found. Cannot parse Trivy results."
        return 0
    fi
    
    log_info "Parsing Trivy results..."
    
    local critical=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' "$json_file")
    local high=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH")] | length' "$json_file")
    local medium=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "MEDIUM")] | length' "$json_file")
    local low=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "LOW")] | length' "$json_file")
    
    log_info "Trivy Vulnerability Summary:"
    log_info "  Critical: $critical"
    log_info "  High: $high"
    log_info "  Medium: $medium"
    log_info "  Low: $low"
    
    # Check against thresholds
    local exit_code=0
    
    if [ "$critical" -gt "$MAX_CRITICAL" ]; then
        log_error "Critical vulnerabilities ($critical) exceed threshold ($MAX_CRITICAL)"
        exit_code=1
    fi
    
    if [ "$high" -gt "$MAX_HIGH" ]; then
        log_error "High vulnerabilities ($high) exceed threshold ($MAX_HIGH)"
        exit_code=1
    fi
    
    if [ "$medium" -gt "$MAX_MEDIUM" ]; then
        log_error "Medium vulnerabilities ($medium) exceed threshold ($MAX_MEDIUM)"
        exit_code=1
    fi
    
    return $exit_code
}

# Docker Scout scanning
scan_with_docker_scout() {
    if [ "$DOCKER_SCOUT_ENABLED" != "true" ]; then
        log_info "Skipping Docker Scout scan (disabled)"
        return 0
    fi
    
    log_info "Scanning with Docker Scout..."
    
    local output_file="${REPORT_DIR}/scout/scout-report.json"
    
    # Basic vulnerability scan
    if docker scout cves --format sarif --output "$output_file" "$FULL_IMAGE_NAME"; then
        log_info "Docker Scout scan completed"
        log_info "Report saved to: $output_file"
    else
        log_warn "Docker Scout scan failed or found issues"
    fi
    
    # Policy evaluation (if available)
    if docker scout policy --format json --output "${REPORT_DIR}/scout/scout-policy.json" "$FULL_IMAGE_NAME" 2>/dev/null; then
        log_info "Docker Scout policy evaluation completed"
    fi
}

# Snyk scanning
scan_with_snyk() {
    if [ "$SNYK_ENABLED" != "true" ]; then
        log_info "Skipping Snyk scan (disabled)"
        return 0
    fi
    
    log_info "Scanning with Snyk..."
    
    # Authenticate Snyk
    echo "$SNYK_TOKEN" | snyk auth
    
    local json_output="${REPORT_DIR}/snyk/snyk-report.json"
    local sarif_output="${REPORT_DIR}/snyk/snyk-report.sarif"
    
    # Scan container image
    if snyk container test "$FULL_IMAGE_NAME" --json > "$json_output" 2>&1; then
        log_info "Snyk scan completed successfully"
    else
        log_warn "Snyk found vulnerabilities"
    fi
    
    # Generate SARIF report
    snyk container test "$FULL_IMAGE_NAME" --sarif > "$sarif_output" 2>&1 || true
    
    log_info "Snyk reports saved to: ${REPORT_DIR}/snyk/"
}

# Grype scanning
scan_with_grype() {
    if [ "$GRYPE_ENABLED" != "true" ]; then
        log_info "Skipping Grype scan (disabled)"
        return 0
    fi
    
    log_info "Scanning with Grype..."
    
    local json_output="${REPORT_DIR}/grype/grype-report.json"
    local table_output="${REPORT_DIR}/grype/grype-report.txt"
    
    # Scan image
    grype "$FULL_IMAGE_NAME" -o json > "$json_output"
    grype "$FULL_IMAGE_NAME" -o table > "$table_output"
    
    log_info "Grype scan completed"
    log_info "Reports saved to: ${REPORT_DIR}/grype/"
}

# Generate summary report
generate_summary() {
    log_info "Generating summary report..."
    
    local summary_file="${REPORT_DIR}/security-summary.md"
    
    cat > "$summary_file" << EOF
# Security Scan Summary

**Image:** $FULL_IMAGE_NAME
**Scan Date:** $(date)
**Scanner Version:** Docker Security Scanner v1.0

## Scan Tools Used

EOF
    
    if [ "$TRIVY_ENABLED" = "true" ]; then
        echo "- ✅ Trivy (vulnerability, secrets, config)" >> "$summary_file"
    fi
    
    if [ "$DOCKER_SCOUT_ENABLED" = "true" ]; then
        echo "- ✅ Docker Scout (vulnerability, policy)" >> "$summary_file"
    fi
    
    if [ "$HADOLINT_ENABLED" = "true" ]; then
        echo "- ✅ Hadolint (Dockerfile linting)" >> "$summary_file"
    fi
    
    if [ "$SNYK_ENABLED" = "true" ]; then
        echo "- ✅ Snyk (vulnerability)" >> "$summary_file"
    fi
    
    if [ "$GRYPE_ENABLED" = "true" ]; then
        echo "- ✅ Grype (vulnerability)" >> "$summary_file"
    fi
    
    cat >> "$summary_file" << EOF

## Reports Location

All detailed reports are available in: \`$REPORT_DIR\`

### Report Files
EOF
    
    find "$REPORT_DIR" -name "*.json" -o -name "*.txt" -o -name "*.sarif" | sort | while read -r file; do
        local relative_path=$(basename "$file")
        local tool_dir=$(basename "$(dirname "$file")")
        echo "- $tool_dir/$relative_path" >> "$summary_file"
    done
    
    cat >> "$summary_file" << EOF

## Thresholds

- Critical: ≤ $MAX_CRITICAL
- High: ≤ $MAX_HIGH  
- Medium: ≤ $MAX_MEDIUM

## Next Steps

1. Review detailed reports for each tool
2. Prioritize fixing critical and high severity vulnerabilities
3. Update base images and dependencies
4. Re-run scan to verify fixes

EOF
    
    log_info "Summary report generated: $summary_file"
}

# Main execution
main() {
    log_info "Starting Docker security scan..."
    log_info "Image: $FULL_IMAGE_NAME"
    log_info "Output: $REPORT_DIR"
    
    check_tools
    setup_output
    check_image
    
    local scan_exit_code=0
    
    # Run scans
    scan_dockerfile || true
    scan_with_trivy || scan_exit_code=1
    scan_with_docker_scout || true
    scan_with_snyk || true
    scan_with_grype || true
    
    # Generate summary
    generate_summary
    
    log_info "Security scan completed"
    log_info "Reports available in: $REPORT_DIR"
    
    if [ $scan_exit_code -ne 0 ]; then
        log_error "Security scan found issues that exceed configured thresholds"
        exit 1
    else
        log_info "Security scan passed all configured thresholds"
    fi
}

# Parse arguments and run
parse_args "$@"
main