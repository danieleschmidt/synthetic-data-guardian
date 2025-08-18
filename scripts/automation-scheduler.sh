#!/bin/bash

# =============================================================================
# Automation Scheduler for Synthetic Data Guardian
# =============================================================================
# This script orchestrates various automated tasks:
# - Metrics collection and reporting
# - Security scanning and alerting
# - Code quality monitoring
# - Performance benchmarking
# - Repository maintenance

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
REPORTS_DIR="$PROJECT_ROOT/reports"
CONFIG_DIR="$PROJECT_ROOT/automation"

# Ensure directories exist
mkdir -p "$LOG_DIR" "$REPORTS_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]${NC} $1" | tee -a "$LOG_DIR/automation.log"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] [SUCCESS]${NC} $1" | tee -a "$LOG_DIR/automation.log"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] [WARNING]${NC} $1" | tee -a "$LOG_DIR/automation.log"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR]${NC} $1" | tee -a "$LOG_DIR/automation.log"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Send notification (placeholder for actual implementation)
send_notification() {
    local level="$1"
    local message="$2"
    
    # Here you would integrate with actual notification systems
    log_info "NOTIFICATION [$level]: $message"
    
    # Example integrations:
    # - Slack webhook
    # - Email via sendmail/SMTP
    # - PagerDuty API
    # - Microsoft Teams webhook
    # - Discord webhook
    
    if [[ -n "${SLACK_WEBHOOK_URL:-}" && "$level" != "info" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🤖 Synthetic Guardian Automation [$level]: $message\"}" \
            "$SLACK_WEBHOOK_URL" >/dev/null 2>&1 || true
    fi
}

# Collect project metrics
collect_metrics() {
    log_info "Starting metrics collection..."
    
    cd "$PROJECT_ROOT"
    
    # Run the Python metrics collector
    if command_exists python3 && [[ -f "$SCRIPT_DIR/automated-metrics-collector.py" ]]; then
        python3 "$SCRIPT_DIR/automated-metrics-collector.py" \
            --config "$CONFIG_DIR/metrics-config.json" \
            --output "$REPORTS_DIR/metrics-$(date +%Y%m%d).json" || {
            log_error "Metrics collection failed"
            send_notification "error" "Metrics collection failed"
            return 1
        }
        log_success "Metrics collection completed"
    else
        log_warning "Python metrics collector not available"
    fi
    
    # Collect Git statistics
    if command_exists git; then
        {
            echo "Git Statistics Report - $(date)"
            echo "=================================="
            echo
            echo "Repository: $(git config --get remote.origin.url)"
            echo "Current branch: $(git branch --show-current)"
            echo "Latest commit: $(git log -1 --pretty=format:'%h - %s (%cr) <%an>')"
            echo
            echo "Commit activity (last 30 days):"
            git log --since="30 days ago" --pretty=format:'%cd %an' --date=short | \
                sort | uniq -c | sort -nr | head -10
            echo
            echo "File changes (last 30 days):"
            git log --since="30 days ago" --name-only --pretty=format: | \
                sort | uniq -c | sort -nr | head -10
            echo
            echo "Contributors (last 30 days):"
            git log --since="30 days ago" --pretty=format:'%an' | sort | uniq -c | sort -nr
        } > "$REPORTS_DIR/git-stats-$(date +%Y%m%d).txt"
        
        log_success "Git statistics collected"
    fi
}

# Run security scans
run_security_scans() {
    log_info "Starting security scans..."
    
    cd "$PROJECT_ROOT"
    
    local security_issues=0
    
    # NPM audit
    if [[ -f "package.json" ]] && command_exists npm; then
        log_info "Running npm audit..."
        if ! npm audit --audit-level=moderate > "$REPORTS_DIR/npm-audit-$(date +%Y%m%d).json" 2>&1; then
            log_warning "NPM audit found issues"
            ((security_issues++))
        fi
    fi
    
    # Python safety check
    if [[ -f "requirements.txt" ]] && command_exists safety; then
        log_info "Running Python safety check..."
        if ! safety check --json > "$REPORTS_DIR/safety-check-$(date +%Y%m%d).json" 2>&1; then
            log_warning "Python safety check found issues"
            ((security_issues++))
        fi
    fi
    
    # Bandit security scan for Python
    if [[ -d "src" ]] && command_exists bandit; then
        log_info "Running Bandit security scan..."
        if ! bandit -r src/ -f json -o "$REPORTS_DIR/bandit-scan-$(date +%Y%m%d).json" 2>/dev/null; then
            log_warning "Bandit scan found issues"
            ((security_issues++))
        fi
    fi
    
    # Docker image security scan with Trivy
    if command_exists docker && command_exists trivy; then
        log_info "Running Trivy container scan..."
        if docker images | grep -q synthetic-data-guardian; then
            trivy image --format json --output "$REPORTS_DIR/trivy-scan-$(date +%Y%m%d).json" \
                synthetic-data-guardian:latest || {
                log_warning "Trivy scan found issues"
                ((security_issues++))
            }
        fi
    fi
    
    # Secret scanning with git-secrets
    if command_exists git-secrets; then
        log_info "Running secret scanning..."
        if ! git secrets --scan 2>&1 | tee "$REPORTS_DIR/secrets-scan-$(date +%Y%m%d).txt"; then
            log_warning "Secret scanning found issues"
            ((security_issues++))
        fi
    fi
    
    if [[ $security_issues -gt 0 ]]; then
        log_warning "Security scans completed with $security_issues issues found"
        send_notification "warning" "Security scans found $security_issues issues"
    else
        log_success "Security scans completed - no issues found"
    fi
}

# Monitor code quality
monitor_code_quality() {
    log_info "Starting code quality monitoring..."
    
    cd "$PROJECT_ROOT"
    
    local quality_issues=0
    
    # Run linting
    if [[ -f "package.json" ]] && command_exists npm; then
        log_info "Running ESLint..."
        if ! npm run lint > "$REPORTS_DIR/eslint-$(date +%Y%m%d).txt" 2>&1; then
            log_warning "ESLint found issues"
            ((quality_issues++))
        fi
        
        log_info "Running Prettier check..."
        if ! npm run format:check > "$REPORTS_DIR/prettier-$(date +%Y%m%d).txt" 2>&1; then
            log_warning "Prettier formatting issues found"
            ((quality_issues++))
        fi
        
        log_info "Running TypeScript check..."
        if ! npm run typecheck > "$REPORTS_DIR/typecheck-$(date +%Y%m%d).txt" 2>&1; then
            log_warning "TypeScript issues found"
            ((quality_issues++))
        fi
    fi
    
    # Python code quality
    if [[ -d "src" ]] && command_exists python3; then
        if command_exists black; then
            log_info "Running Black format check..."
            if ! black --check src/ > "$REPORTS_DIR/black-check-$(date +%Y%m%d).txt" 2>&1; then
                log_warning "Black formatting issues found"
                ((quality_issues++))
            fi
        fi
        
        if command_exists flake8; then
            log_info "Running Flake8..."
            if ! flake8 src/ > "$REPORTS_DIR/flake8-$(date +%Y%m%d).txt" 2>&1; then
                log_warning "Flake8 issues found"
                ((quality_issues++))
            fi
        fi
        
        if command_exists mypy; then
            log_info "Running MyPy type checking..."
            if ! mypy src/ > "$REPORTS_DIR/mypy-$(date +%Y%m%d).txt" 2>&1; then
                log_warning "MyPy type issues found"
                ((quality_issues++))
            fi
        fi
    fi
    
    if [[ $quality_issues -gt 0 ]]; then
        log_warning "Code quality monitoring completed with $quality_issues issues found"
        send_notification "warning" "Code quality monitoring found $quality_issues issues"
    else
        log_success "Code quality monitoring completed - no issues found"
    fi
}

# Run performance benchmarks
run_performance_benchmarks() {
    log_info "Starting performance benchmarks..."
    
    cd "$PROJECT_ROOT"
    
    # Run load tests if k6 is available
    if command_exists k6 && [[ -f "tests/performance/load-test.js" ]]; then
        log_info "Running k6 load tests..."
        k6 run tests/performance/load-test.js \
            --out json="$REPORTS_DIR/performance-$(date +%Y%m%d).json" || {
            log_warning "Performance tests showed degradation"
            send_notification "warning" "Performance tests showed degradation"
        }
    fi
    
    # Memory usage check
    if command_exists ps && command_exists pgrep; then
        log_info "Checking memory usage..."
        {
            echo "Memory Usage Report - $(date)"
            echo "============================="
            echo
            ps aux --sort=-%mem | head -20
        } > "$REPORTS_DIR/memory-usage-$(date +%Y%m%d).txt"
    fi
    
    # Disk usage check
    if command_exists df; then
        log_info "Checking disk usage..."
        {
            echo "Disk Usage Report - $(date)"
            echo "=========================="
            echo
            df -h
            echo
            echo "Largest directories in project:"
            du -sh ./* 2>/dev/null | sort -rh | head -10
        } > "$REPORTS_DIR/disk-usage-$(date +%Y%m%d).txt"
    fi
    
    log_success "Performance benchmarks completed"
}

# Repository maintenance
repository_maintenance() {
    log_info "Starting repository maintenance..."
    
    cd "$PROJECT_ROOT"
    
    # Clean up old logs
    if [[ -d "$LOG_DIR" ]]; then
        find "$LOG_DIR" -name "*.log" -mtime +30 -delete 2>/dev/null || true
        log_info "Cleaned up old log files"
    fi
    
    # Clean up old reports
    if [[ -d "$REPORTS_DIR" ]]; then
        find "$REPORTS_DIR" -name "*.json" -mtime +90 -delete 2>/dev/null || true
        find "$REPORTS_DIR" -name "*.txt" -mtime +90 -delete 2>/dev/null || true
        log_info "Cleaned up old report files"
    fi
    
    # Git maintenance
    if command_exists git; then
        log_info "Running git maintenance..."
        git gc --auto || true
        git prune || true
    fi
    
    # Docker cleanup
    if command_exists docker; then
        log_info "Cleaning up Docker resources..."
        docker system prune -f || true
    fi
    
    # Node modules cleanup (if needed)
    if [[ -d "node_modules" ]] && [[ -f "package-lock.json" ]]; then
        # Check if package-lock.json is newer than node_modules
        if [[ "package-lock.json" -nt "node_modules" ]]; then
            log_info "Package lock file is newer, running npm ci..."
            npm ci || log_warning "npm ci failed"
        fi
    fi
    
    log_success "Repository maintenance completed"
}

# Generate daily summary report
generate_summary_report() {
    log_info "Generating daily summary report..."
    
    local report_file="$REPORTS_DIR/daily-summary-$(date +%Y%m%d).md"
    
    {
        echo "# Daily Summary Report"
        echo "**Date:** $(date +'%Y-%m-%d %H:%M:%S')"
        echo "**Repository:** synthetic-data-guardian"
        echo
        
        echo "## Automation Tasks Completed"
        echo "- ✅ Metrics collection"
        echo "- ✅ Security scanning"
        echo "- ✅ Code quality monitoring"
        echo "- ✅ Performance benchmarking"
        echo "- ✅ Repository maintenance"
        echo
        
        echo "## Recent Git Activity"
        if command_exists git; then
            echo "**Latest Commits:**"
            git log --oneline -5 | sed 's/^/- /'
            echo
            
            echo "**Active Contributors (last 7 days):**"
            git log --since="7 days ago" --pretty=format:'%an' | sort | uniq -c | sort -nr | head -5 | sed 's/^/- /'
            echo
        fi
        
        echo "## File Reports Generated"
        echo "The following reports were generated today:"
        find "$REPORTS_DIR" -name "*$(date +%Y%m%d)*" -type f | sort | sed 's/^/- /'
        echo
        
        echo "## System Health"
        if command_exists df; then
            echo "**Disk Usage:**"
            df -h | grep -E '^/dev/' | sed 's/^/- /'
            echo
        fi
        
        if command_exists free; then
            echo "**Memory Usage:**"
            free -h | sed 's/^/- /'
            echo
        fi
        
        echo "## Next Scheduled Run"
        echo "Next automation run scheduled for tomorrow at the same time."
        echo
        
        echo "---"
        echo "*Generated by Synthetic Data Guardian Automation System*"
        
    } > "$report_file"
    
    log_success "Daily summary report generated: $report_file"
}

# Main automation workflow
run_automation() {
    local task_type="${1:-full}"
    
    log_info "Starting automation workflow: $task_type"
    send_notification "info" "Starting automation workflow: $task_type"
    
    case "$task_type" in
        "metrics")
            collect_metrics
            ;;
        "security")
            run_security_scans
            ;;
        "quality")
            monitor_code_quality
            ;;
        "performance")
            run_performance_benchmarks
            ;;
        "maintenance")
            repository_maintenance
            ;;
        "daily"|"full")
            collect_metrics
            run_security_scans
            monitor_code_quality
            run_performance_benchmarks
            repository_maintenance
            generate_summary_report
            ;;
        "hourly")
            collect_metrics
            monitor_code_quality
            ;;
        "weekly")
            collect_metrics
            run_security_scans
            monitor_code_quality
            run_performance_benchmarks
            repository_maintenance
            generate_summary_report
            ;;
        *)
            log_error "Unknown task type: $task_type"
            echo "Usage: $0 [metrics|security|quality|performance|maintenance|daily|hourly|weekly|full]"
            exit 1
            ;;
    esac
    
    log_success "Automation workflow completed: $task_type"
    send_notification "success" "Automation workflow completed: $task_type"
}

# Trap to ensure cleanup on exit
cleanup() {
    log_info "Automation script cleanup"
}
trap cleanup EXIT

# Main entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed directly
    task_type="${1:-daily}"
    run_automation "$task_type"
fi