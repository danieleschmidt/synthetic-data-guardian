#!/usr/bin/env python3

"""
Automated Metrics Collector for Synthetic Data Guardian

This script collects and aggregates metrics from various sources:
- GitHub API (commits, PRs, issues, contributors)
- CI/CD systems (build times, test results)
- Security scanners (vulnerability counts, scores)
- Code quality tools (coverage, complexity)
- Monitoring systems (performance, uptime)
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import requests
import argparse


class MetricsCollector:
    def __init__(self, config_path: str = "automation/metrics-config.json"):
        """Initialize the metrics collector with configuration."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.session = requests.Session()
        self.metrics_data = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise Exception(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON in configuration file: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/metrics-collector.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def collect_github_metrics(self) -> Dict[str, Any]:
        """Collect metrics from GitHub API."""
        if not self.config.get('github', {}).get('enabled', False):
            return {}
        
        self.logger.info("Collecting GitHub metrics...")
        
        github_token = os.getenv('GITHUB_TOKEN')
        if not github_token:
            self.logger.warning("GITHUB_TOKEN not found, skipping GitHub metrics")
            return {}
        
        headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        repo = self.config['github']['repository']
        base_url = f"https://api.github.com/repos/{repo}"
        
        metrics = {}
        
        try:
            # Repository basic info
            repo_response = self.session.get(f"{base_url}", headers=headers)
            repo_response.raise_for_status()
            repo_data = repo_response.json()
            
            metrics.update({
                'stars': repo_data.get('stargazers_count', 0),
                'forks': repo_data.get('forks_count', 0),
                'watchers': repo_data.get('watchers_count', 0),
                'open_issues': repo_data.get('open_issues_count', 0),
                'size_kb': repo_data.get('size', 0),
                'default_branch': repo_data.get('default_branch', 'main')
            })
            
            # Commits in the last 30 days
            since_date = (datetime.now() - timedelta(days=30)).isoformat()
            commits_response = self.session.get(
                f"{base_url}/commits",
                headers=headers,
                params={'since': since_date, 'per_page': 100}
            )
            commits_response.raise_for_status()
            commits_data = commits_response.json()
            metrics['commits_last_30_days'] = len(commits_data)
            
            # Active contributors (unique committers in last 30 days)
            contributors = set()
            for commit in commits_data:
                if commit.get('author') and commit['author'].get('login'):
                    contributors.add(commit['author']['login'])
            metrics['active_contributors'] = len(contributors)
            
            # Pull requests
            prs_response = self.session.get(
                f"{base_url}/pulls",
                headers=headers,
                params={'state': 'all', 'per_page': 100}
            )
            prs_response.raise_for_status()
            prs_data = prs_response.json()
            
            open_prs = [pr for pr in prs_data if pr['state'] == 'open']
            closed_prs = [pr for pr in prs_data if pr['state'] == 'closed']
            
            metrics.update({
                'open_pull_requests': len(open_prs),
                'closed_pull_requests_last_100': len(closed_prs)
            })
            
            # Calculate average PR merge time (for merged PRs)
            merge_times = []
            for pr in closed_prs:
                if pr.get('merged_at'):
                    created = datetime.fromisoformat(pr['created_at'].replace('Z', '+00:00'))
                    merged = datetime.fromisoformat(pr['merged_at'].replace('Z', '+00:00'))
                    merge_time_hours = (merged - created).total_seconds() / 3600
                    merge_times.append(merge_time_hours)
            
            if merge_times:
                metrics['average_pr_merge_time_hours'] = sum(merge_times) / len(merge_times)
            else:
                metrics['average_pr_merge_time_hours'] = 0
                
            # Issues
            issues_response = self.session.get(
                f"{base_url}/issues",
                headers=headers,
                params={'state': 'open', 'per_page': 100}
            )
            issues_response.raise_for_status()
            issues_data = issues_response.json()
            
            # Filter out pull requests (GitHub API includes PRs in issues)
            actual_issues = [issue for issue in issues_data if not issue.get('pull_request')]
            metrics['open_issues_actual'] = len(actual_issues)
            
            self.logger.info(f"Collected GitHub metrics: {len(metrics)} metrics")
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error collecting GitHub metrics: {e}")
            
        return metrics
    
    def collect_ci_metrics(self) -> Dict[str, Any]:
        """Collect CI/CD metrics from GitHub Actions."""
        self.logger.info("Collecting CI/CD metrics...")
        
        github_token = os.getenv('GITHUB_TOKEN')
        if not github_token:
            return {}
        
        headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        repo = self.config['github']['repository']
        base_url = f"https://api.github.com/repos/{repo}"
        
        metrics = {}
        
        try:
            # Get recent workflow runs
            workflows_response = self.session.get(
                f"{base_url}/actions/runs",
                headers=headers,
                params={'per_page': 50}
            )
            workflows_response.raise_for_status()
            workflows_data = workflows_response.json()
            
            if 'workflow_runs' in workflows_data:
                runs = workflows_data['workflow_runs']
                
                # Calculate success rate
                successful_runs = [run for run in runs if run['conclusion'] == 'success']
                total_runs = len(runs)
                
                if total_runs > 0:
                    metrics['build_success_rate'] = (len(successful_runs) / total_runs) * 100
                else:
                    metrics['build_success_rate'] = 0
                
                # Calculate average build time (for completed runs)
                build_times = []
                for run in runs:
                    if run.get('conclusion') and run.get('created_at') and run.get('updated_at'):
                        created = datetime.fromisoformat(run['created_at'].replace('Z', '+00:00'))
                        updated = datetime.fromisoformat(run['updated_at'].replace('Z', '+00:00'))
                        duration_minutes = (updated - created).total_seconds() / 60
                        build_times.append(duration_minutes)
                
                if build_times:
                    metrics['average_build_time_minutes'] = sum(build_times) / len(build_times)
                else:
                    metrics['average_build_time_minutes'] = 0
                
                # Recent build status
                if runs:
                    latest_run = runs[0]
                    metrics.update({
                        'latest_build_status': latest_run.get('conclusion', 'unknown'),
                        'latest_build_branch': latest_run.get('head_branch', 'unknown'),
                        'latest_build_commit': latest_run.get('head_sha', '')[:8]
                    })
            
            self.logger.info(f"Collected CI/CD metrics: {len(metrics)} metrics")
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error collecting CI/CD metrics: {e}")
            
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics from various sources."""
        self.logger.info("Collecting security metrics...")
        
        metrics = {}
        
        # GitHub Security Advisory
        github_token = os.getenv('GITHUB_TOKEN')
        if github_token:
            headers = {
                'Authorization': f'token {github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            repo = self.config['github']['repository']
            base_url = f"https://api.github.com/repos/{repo}"
            
            try:
                # Dependabot alerts
                alerts_response = self.session.get(
                    f"{base_url}/dependabot/alerts",
                    headers=headers
                )
                
                if alerts_response.status_code == 200:
                    alerts_data = alerts_response.json()
                    open_alerts = [alert for alert in alerts_data if alert['state'] == 'open']
                    
                    # Count by severity
                    severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
                    for alert in open_alerts:
                        severity = alert.get('security_advisory', {}).get('severity', 'low')
                        if severity in severity_counts:
                            severity_counts[severity] += 1
                    
                    metrics.update({
                        'dependabot_alerts_total': len(open_alerts),
                        'dependabot_alerts_critical': severity_counts['critical'],
                        'dependabot_alerts_high': severity_counts['high'],
                        'dependabot_alerts_medium': severity_counts['medium'],
                        'dependabot_alerts_low': severity_counts['low']
                    })
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Could not fetch Dependabot alerts: {e}")
        
        # If we have local security scan results, parse them
        security_files = [
            'security-scan-results.json',
            'snyk-results.json',
            'safety-results.json'
        ]
        
        for file_path in security_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        security_data = json.load(f)
                        # Parse security data based on tool format
                        # This would need to be customized for each tool's output format
                        metrics[f'{file_path}_last_scan'] = datetime.now().isoformat()
                except Exception as e:
                    self.logger.warning(f"Could not parse {file_path}: {e}")
        
        self.logger.info(f"Collected security metrics: {len(metrics)} metrics")
        return metrics
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        self.logger.info("Collecting code quality metrics...")
        
        metrics = {}
        
        # Check for coverage reports
        coverage_files = [
            'coverage/lcov.info',
            'coverage.xml',
            'htmlcov/index.html'
        ]
        
        for file_path in coverage_files:
            if os.path.exists(file_path):
                metrics['coverage_report_exists'] = True
                metrics['coverage_report_path'] = file_path
                break
        else:
            metrics['coverage_report_exists'] = False
        
        # Check for test results
        test_result_files = [
            'test-results.xml',
            'junit-results.xml',
            'pytest-results.xml'
        ]
        
        for file_path in test_result_files:
            if os.path.exists(file_path):
                metrics['test_results_exist'] = True
                metrics['test_results_path'] = file_path
                break
        else:
            metrics['test_results_exist'] = False
        
        # Count test files
        test_file_count = 0
        for root, dirs, files in os.walk('.'):
            for file in files:
                if (file.startswith('test_') and file.endswith('.py')) or \
                   (file.endswith('.test.js') or file.endswith('.test.ts')):
                    test_file_count += 1
        
        metrics['test_file_count'] = test_file_count
        
        # Count source files
        source_file_count = 0
        for root, dirs, files in os.walk('src'):
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.jsx', '.tsx')):
                    source_file_count += 1
        
        metrics['source_file_count'] = source_file_count
        
        # Calculate test-to-source ratio
        if source_file_count > 0:
            metrics['test_to_source_ratio'] = test_file_count / source_file_count
        else:
            metrics['test_to_source_ratio'] = 0
        
        self.logger.info(f"Collected code quality metrics: {len(metrics)} metrics")
        return metrics
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        self.logger.info("Starting comprehensive metrics collection...")
        
        all_metrics = {
            'collection_timestamp': datetime.now().isoformat(),
            'collector_version': '1.0.0'
        }
        
        # Collect from all sources
        github_metrics = self.collect_github_metrics()
        ci_metrics = self.collect_ci_metrics()
        security_metrics = self.collect_security_metrics()
        quality_metrics = self.collect_code_quality_metrics()
        
        # Combine all metrics
        all_metrics.update({
            'github': github_metrics,
            'ci_cd': ci_metrics,
            'security': security_metrics,
            'code_quality': quality_metrics
        })
        
        # Calculate composite scores
        all_metrics['composite_scores'] = self._calculate_composite_scores(all_metrics)
        
        self.logger.info("Metrics collection completed successfully")
        return all_metrics
    
    def _calculate_composite_scores(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate composite scores from individual metrics."""
        scores = {}
        
        # Development Health Score (0-100)
        github_metrics = metrics.get('github', {})
        ci_metrics = metrics.get('ci_cd', {})
        
        dev_health_factors = []
        
        # CI/CD health
        build_success_rate = ci_metrics.get('build_success_rate', 0)
        dev_health_factors.append(min(build_success_rate, 100))
        
        # Code activity (commits in last 30 days, capped at 100 for score calculation)
        commits = github_metrics.get('commits_last_30_days', 0)
        dev_health_factors.append(min(commits * 2, 100))  # 50+ commits = full score
        
        # Community engagement (contributors, PRs)
        contributors = github_metrics.get('active_contributors', 0)
        dev_health_factors.append(min(contributors * 20, 100))  # 5+ contributors = full score
        
        if dev_health_factors:
            scores['development_health'] = sum(dev_health_factors) / len(dev_health_factors)
        
        # Security Health Score (0-100)
        security_metrics = metrics.get('security', {})
        
        # Start with perfect score and deduct for issues
        security_score = 100
        
        critical_alerts = security_metrics.get('dependabot_alerts_critical', 0)
        high_alerts = security_metrics.get('dependabot_alerts_high', 0)
        medium_alerts = security_metrics.get('dependabot_alerts_medium', 0)
        
        # Deduct points for vulnerabilities
        security_score -= critical_alerts * 25  # Critical = -25 points each
        security_score -= high_alerts * 10     # High = -10 points each
        security_score -= medium_alerts * 5    # Medium = -5 points each
        
        scores['security_health'] = max(security_score, 0)
        
        # Quality Health Score (0-100)
        quality_metrics = metrics.get('code_quality', {})
        
        quality_factors = []
        
        # Test coverage (assuming we have it)
        if quality_metrics.get('coverage_report_exists'):
            quality_factors.append(80)  # Assume good coverage if report exists
        else:
            quality_factors.append(40)   # Lower score if no coverage tracking
        
        # Test-to-source ratio
        test_ratio = quality_metrics.get('test_to_source_ratio', 0)
        quality_factors.append(min(test_ratio * 100, 100))  # 1:1 ratio = full score
        
        if quality_factors:
            scores['quality_health'] = sum(quality_factors) / len(quality_factors)
        
        # Overall Health Score
        if scores:
            scores['overall_health'] = sum(scores.values()) / len(scores)
        
        return scores
    
    def save_metrics(self, metrics: Dict[str, Any], output_path: str = None) -> str:
        """Save metrics to file."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"metrics-report-{timestamp}.json"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        self.logger.info(f"Metrics saved to: {output_path}")
        return output_path
    
    def send_alerts(self, metrics: Dict[str, Any]) -> None:
        """Send alerts based on metric thresholds."""
        if not self.config.get('alerts', {}).get('enabled', False):
            return
        
        alerts = []
        composite_scores = metrics.get('composite_scores', {})
        
        # Check thresholds
        thresholds = self.config['alerts']['thresholds']
        
        for score_name, score_value in composite_scores.items():
            if score_name in thresholds.get('critical', {}):
                threshold = thresholds['critical'][score_name]
                if score_value < threshold:
                    alerts.append({
                        'level': 'critical',
                        'metric': score_name,
                        'value': score_value,
                        'threshold': threshold,
                        'message': f"{score_name} is critically low: {score_value:.1f} < {threshold}"
                    })
            elif score_name in thresholds.get('warning', {}):
                threshold = thresholds['warning'][score_name]
                if score_value < threshold:
                    alerts.append({
                        'level': 'warning',
                        'metric': score_name,
                        'value': score_value,
                        'threshold': threshold,
                        'message': f"{score_name} is below warning threshold: {score_value:.1f} < {threshold}"
                    })
        
        # Send alerts if any
        if alerts:
            self.logger.warning(f"Generated {len(alerts)} alerts")
            for alert in alerts:
                self.logger.warning(f"ALERT [{alert['level'].upper()}]: {alert['message']}")
            
            # Here you would integrate with actual alerting systems
            # (Slack, email, PagerDuty, etc.)
        else:
            self.logger.info("No alerts generated - all metrics within thresholds")


def main():
    """Main entry point for the metrics collector."""
    parser = argparse.ArgumentParser(description='Collect project metrics')
    parser.add_argument('--config', default='automation/metrics-config.json',
                       help='Path to configuration file')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--alerts-only', action='store_true',
                       help='Only check alerts, do not collect metrics')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        collector = MetricsCollector(args.config)
        
        if args.alerts_only:
            # Load existing metrics and check alerts
            if os.path.exists('metrics-report.json'):
                with open('metrics-report.json', 'r') as f:
                    metrics = json.load(f)
                collector.send_alerts(metrics)
            else:
                print("No existing metrics file found for alert checking")
        else:
            # Collect all metrics
            metrics = collector.collect_all_metrics()
            
            # Save metrics
            output_path = collector.save_metrics(metrics, args.output)
            
            # Check alerts
            collector.send_alerts(metrics)
            
            print(f"Metrics collection completed. Report saved to: {output_path}")
            
            # Print summary
            composite_scores = metrics.get('composite_scores', {})
            if composite_scores:
                print("\nComposite Scores:")
                for score_name, score_value in composite_scores.items():
                    print(f"  {score_name}: {score_value:.1f}/100")
    
    except Exception as e:
        logging.error(f"Metrics collection failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()