#!/usr/bin/env python3
"""
Synthetic Data Guardian - Metrics Collection Script

Automated metrics collection and reporting for repository health, code quality,
security posture, and performance tracking.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import logging
import requests
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects various metrics for the Synthetic Data Guardian project."""
    
    def __init__(self, repo_path: str = ".", config_path: Optional[str] = None):
        self.repo_path = Path(repo_path)
        self.config = self._load_config(config_path)
        self.metrics = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "github": {
                "enabled": True,
                "repository": "danieleschmidt/synthetic-data-guardian"
            },
            "sonarqube": {
                "enabled": False,
                "url": "",
                "token": ""
            },
            "codecov": {
                "enabled": False,
                "token": ""
            },
            "prometheus": {
                "enabled": False,
                "url": "http://localhost:9090"
            },
            "output": {
                "format": "json",
                "file": "metrics-report.json",
                "include_timestamps": True
            },
            "thresholds": {
                "code_coverage": 80,
                "test_pass_rate": 95,
                "security_score": 85,
                "performance_score": 80
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
                
        return default_config
    
    def run_command(self, command: List[str], cwd: Optional[str] = None) -> Dict[str, Any]:
        """Run a shell command and return the result."""
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.repo_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
    
    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git repository metrics."""
        logger.info("Collecting Git metrics...")
        
        metrics = {
            "repository_health": {},
            "commit_activity": {},
            "branch_info": {}
        }
        
        # Get current branch and commit info
        branch_result = self.run_command(["git", "branch", "--show-current"])
        if branch_result["success"]:
            metrics["branch_info"]["current_branch"] = branch_result["stdout"]
            
        commit_result = self.run_command(["git", "rev-parse", "HEAD"])
        if commit_result["success"]:
            metrics["branch_info"]["latest_commit"] = commit_result["stdout"]
            
        # Get commit count
        commit_count_result = self.run_command(["git", "rev-list", "--count", "HEAD"])
        if commit_count_result["success"]:
            metrics["commit_activity"]["total_commits"] = int(commit_count_result["stdout"])
            
        # Get commits in last 30 days
        recent_commits_result = self.run_command([
            "git", "rev-list", "--count", "--since=30.days", "HEAD"
        ])
        if recent_commits_result["success"]:
            metrics["commit_activity"]["commits_last_30_days"] = int(recent_commits_result["stdout"])
            
        # Get number of contributors
        contributors_result = self.run_command([
            "git", "shortlog", "-sn", "--all"
        ])
        if contributors_result["success"]:
            contributor_count = len(contributors_result["stdout"].split('\n')) if contributors_result["stdout"] else 0
            metrics["repository_health"]["contributor_count"] = contributor_count
            
        # Check if working directory is clean
        status_result = self.run_command(["git", "status", "--porcelain"])
        metrics["repository_health"]["working_directory_clean"] = status_result["success"] and not status_result["stdout"]
        
        return metrics
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        logger.info("Collecting code quality metrics...")
        
        metrics = {
            "lines_of_code": {},
            "file_statistics": {},
            "complexity": {}
        }
        
        # Count lines of code by file type
        file_types = {
            "python": ["*.py"],
            "typescript": ["*.ts", "*.tsx"],
            "javascript": ["*.js", "*.jsx"],
            "yaml": ["*.yml", "*.yaml"],
            "json": ["*.json"],
            "markdown": ["*.md"],
            "dockerfile": ["Dockerfile*", "*.dockerfile"]
        }
        
        for lang, patterns in file_types.items():
            total_lines = 0
            file_count = 0
            
            for pattern in patterns:
                find_result = self.run_command([
                    "find", ".", "-name", pattern, "-type", "f", 
                    "!", "-path", "./node_modules/*",
                    "!", "-path", "./.git/*",
                    "!", "-path", "./venv/*",
                    "!", "-path", "./__pycache__/*"
                ])
                
                if find_result["success"] and find_result["stdout"]:
                    files = find_result["stdout"].split('\n')
                    file_count += len(files)
                    
                    for file_path in files:
                        if file_path.strip():
                            wc_result = self.run_command(["wc", "-l", file_path])
                            if wc_result["success"]:
                                lines = int(wc_result["stdout"].split()[0])
                                total_lines += lines
            
            metrics["lines_of_code"][lang] = {
                "lines": total_lines,
                "files": file_count
            }
        
        # Calculate total LOC
        total_lines = sum(lang_data["lines"] for lang_data in metrics["lines_of_code"].values())
        total_files = sum(lang_data["files"] for lang_data in metrics["lines_of_code"].values())
        
        metrics["file_statistics"]["total_lines"] = total_lines
        metrics["file_statistics"]["total_files"] = total_files
        metrics["file_statistics"]["average_lines_per_file"] = total_lines / total_files if total_files > 0 else 0
        
        return metrics
    
    def collect_test_metrics(self) -> Dict[str, Any]:
        """Collect test coverage and execution metrics."""
        logger.info("Collecting test metrics...")
        
        metrics = {
            "coverage": {},
            "test_execution": {},
            "test_files": {}
        }
        
        # Count test files
        test_patterns = ["*test*.py", "*test*.js", "*test*.ts", "*.test.*", "*.spec.*"]
        total_test_files = 0
        
        for pattern in test_patterns:
            find_result = self.run_command([
                "find", ".", "-name", pattern, "-type", "f",
                "!", "-path", "./node_modules/*",
                "!", "-path", "./.git/*"
            ])
            
            if find_result["success"] and find_result["stdout"]:
                test_files = [f for f in find_result["stdout"].split('\n') if f.strip()]
                total_test_files += len(test_files)
        
        metrics["test_files"]["total_test_files"] = total_test_files
        
        # Try to get coverage from jest if available
        if (self.repo_path / "package.json").exists():
            coverage_result = self.run_command(["npm", "run", "test:coverage", "--silent"])
            if coverage_result["success"]:
                # Parse coverage output (implementation depends on test framework)
                metrics["coverage"]["jest_available"] = True
            else:
                metrics["coverage"]["jest_available"] = False
        
        # Try to get Python coverage if available
        if (self.repo_path / "pytest.ini").exists() or (self.repo_path / "pyproject.toml").exists():
            coverage_result = self.run_command(["python", "-m", "pytest", "--cov", "--cov-report=json", "--tb=no", "-q"])
            if coverage_result["success"]:
                coverage_file = self.repo_path / "coverage.json"
                if coverage_file.exists():
                    with open(coverage_file, 'r') as f:
                        coverage_data = json.load(f)
                        metrics["coverage"]["python_coverage"] = coverage_data.get("totals", {}).get("percent_covered", 0)
            else:
                metrics["coverage"]["python_coverage"] = None
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security-related metrics."""
        logger.info("Collecting security metrics...")
        
        metrics = {
            "dependencies": {},
            "secrets": {},
            "vulnerabilities": {}
        }
        
        # Check for dependency files
        dependency_files = [
            "package.json", "package-lock.json", "yarn.lock",
            "requirements.txt", "requirements-dev.txt", "Pipfile", "Pipfile.lock",
            "pyproject.toml", "poetry.lock", "Cargo.toml", "Cargo.lock",
            "go.mod", "go.sum"
        ]
        
        found_dependencies = []
        for dep_file in dependency_files:
            if (self.repo_path / dep_file).exists():
                found_dependencies.append(dep_file)
        
        metrics["dependencies"]["dependency_files"] = found_dependencies
        
        # Run npm audit if package.json exists
        if "package.json" in found_dependencies:
            audit_result = self.run_command(["npm", "audit", "--json"])
            if audit_result["success"]:
                try:
                    audit_data = json.loads(audit_result["stdout"])
                    metrics["vulnerabilities"]["npm_audit"] = {
                        "total_vulnerabilities": audit_data.get("metadata", {}).get("vulnerabilities", {}).get("total", 0),
                        "critical": audit_data.get("metadata", {}).get("vulnerabilities", {}).get("critical", 0),
                        "high": audit_data.get("metadata", {}).get("vulnerabilities", {}).get("high", 0),
                        "moderate": audit_data.get("metadata", {}).get("vulnerabilities", {}).get("moderate", 0),
                        "low": audit_data.get("metadata", {}).get("vulnerabilities", {}).get("low", 0)
                    }
                except json.JSONDecodeError:
                    metrics["vulnerabilities"]["npm_audit"] = {"error": "Failed to parse audit output"}
        
        # Check for common security files
        security_files = [
            ".github/SECURITY.md", "SECURITY.md", "security.md",
            ".github/dependabot.yml", ".dependabot/config.yml",
            ".snyk", "renovate.json", ".renovaterc"
        ]
        
        found_security_files = []
        for sec_file in security_files:
            if (self.repo_path / sec_file).exists():
                found_security_files.append(sec_file)
        
        metrics["dependencies"]["security_files"] = found_security_files
        
        # Basic secret scanning (look for common patterns)
        secret_patterns = [
            "password", "secret", "token", "key", "api_key",
            "private_key", "access_token", "auth_token"
        ]
        
        potential_secrets = 0
        for pattern in secret_patterns:
            grep_result = self.run_command([
                "grep", "-r", "-i", "--include=*.py", "--include=*.js", 
                "--include=*.ts", "--include=*.json", "--include=*.yml",
                "--exclude-dir=node_modules", "--exclude-dir=.git",
                "--exclude-dir=venv", pattern, "."
            ])
            if grep_result["success"] and grep_result["stdout"]:
                # Count lines, not files
                lines = grep_result["stdout"].split('\n')
                potential_secrets += len([line for line in lines if line.strip()])
        
        metrics["secrets"]["potential_secret_references"] = potential_secrets
        
        return metrics
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance-related metrics."""
        logger.info("Collecting performance metrics...")
        
        metrics = {
            "build_time": {},
            "bundle_size": {},
            "dependencies": {}
        }
        
        # Measure build time if possible
        if (self.repo_path / "package.json").exists():
            start_time = datetime.now()
            build_result = self.run_command(["npm", "run", "build", "--silent"])
            build_time = (datetime.now() - start_time).total_seconds()
            
            metrics["build_time"]["npm_build"] = {
                "success": build_result["success"],
                "duration_seconds": build_time
            }
            
            # Check for build output size
            dist_dir = self.repo_path / "dist"
            build_dir = self.repo_path / "build"
            
            for build_output_dir in [dist_dir, build_dir]:
                if build_output_dir.exists():
                    du_result = self.run_command(["du", "-sb", str(build_output_dir)])
                    if du_result["success"]:
                        size_bytes = int(du_result["stdout"].split()[0])
                        metrics["bundle_size"][build_output_dir.name] = {
                            "size_bytes": size_bytes,
                            "size_mb": round(size_bytes / (1024 * 1024), 2)
                        }
        
        # Count dependencies
        if (self.repo_path / "package.json").exists():
            with open(self.repo_path / "package.json", 'r') as f:
                package_data = json.load(f)
                
            metrics["dependencies"]["npm"] = {
                "production": len(package_data.get("dependencies", {})),
                "development": len(package_data.get("devDependencies", {})),
                "total": len(package_data.get("dependencies", {})) + len(package_data.get("devDependencies", {}))
            }
        
        if (self.repo_path / "requirements.txt").exists():
            req_result = self.run_command(["wc", "-l", "requirements.txt"])
            if req_result["success"]:
                metrics["dependencies"]["python"] = {
                    "requirements_txt": int(req_result["stdout"].split()[0])
                }
        
        return metrics
    
    def collect_github_metrics(self) -> Dict[str, Any]:
        """Collect GitHub-specific metrics using GitHub API."""
        logger.info("Collecting GitHub metrics...")
        
        metrics = {
            "repository": {},
            "issues": {},
            "pull_requests": {},
            "releases": {}
        }
        
        if not self.config["github"]["enabled"]:
            logger.info("GitHub metrics collection disabled in config")
            return metrics
        
        github_token = os.getenv("GITHUB_TOKEN")
        repo = self.config["github"]["repository"]
        
        if not github_token:
            logger.warning("GITHUB_TOKEN not set, skipping GitHub API metrics")
            return metrics
        
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            # Repository info
            repo_response = requests.get(f"https://api.github.com/repos/{repo}", headers=headers)
            if repo_response.status_code == 200:
                repo_data = repo_response.json()
                metrics["repository"] = {
                    "stars": repo_data.get("stargazers_count", 0),
                    "forks": repo_data.get("forks_count", 0),
                    "watchers": repo_data.get("watchers_count", 0),
                    "open_issues": repo_data.get("open_issues_count", 0),
                    "size_kb": repo_data.get("size", 0),
                    "language": repo_data.get("language", ""),
                    "created_at": repo_data.get("created_at", ""),
                    "updated_at": repo_data.get("updated_at", "")
                }
            
            # Issues
            issues_response = requests.get(
                f"https://api.github.com/repos/{repo}/issues?state=all&per_page=100",
                headers=headers
            )
            if issues_response.status_code == 200:
                issues_data = issues_response.json()
                open_issues = [issue for issue in issues_data if issue.get("state") == "open" and not issue.get("pull_request")]
                closed_issues = [issue for issue in issues_data if issue.get("state") == "closed" and not issue.get("pull_request")]
                
                metrics["issues"] = {
                    "open": len(open_issues),
                    "closed": len(closed_issues),
                    "total": len(open_issues) + len(closed_issues)
                }
            
            # Pull requests
            prs_response = requests.get(
                f"https://api.github.com/repos/{repo}/pulls?state=all&per_page=100",
                headers=headers
            )
            if prs_response.status_code == 200:
                prs_data = prs_response.json()
                open_prs = [pr for pr in prs_data if pr.get("state") == "open"]
                closed_prs = [pr for pr in prs_data if pr.get("state") == "closed"]
                
                metrics["pull_requests"] = {
                    "open": len(open_prs),
                    "closed": len(closed_prs),
                    "total": len(open_prs) + len(closed_prs)
                }
            
            # Releases
            releases_response = requests.get(
                f"https://api.github.com/repos/{repo}/releases?per_page=10",
                headers=headers
            )
            if releases_response.status_code == 200:
                releases_data = releases_response.json()
                metrics["releases"] = {
                    "total": len(releases_data),
                    "latest": releases_data[0].get("tag_name", "") if releases_data else ""
                }
                
        except requests.RequestException as e:
            logger.error(f"Failed to fetch GitHub metrics: {e}")
        
        return metrics
    
    def calculate_health_scores(self) -> Dict[str, Any]:
        """Calculate overall health scores based on collected metrics."""
        logger.info("Calculating health scores...")
        
        scores = {
            "overall_health": 0,
            "code_quality": 0,
            "security": 0,
            "maintainability": 0,
            "performance": 0
        }
        
        # Code quality score
        code_quality_factors = []
        
        if "code_quality" in self.metrics:
            total_files = self.metrics["code_quality"]["file_statistics"].get("total_files", 0)
            if total_files > 0:
                code_quality_factors.append(min(100, (total_files / 50) * 100))  # More files = better structure
        
        if "test" in self.metrics:
            test_files = self.metrics["test"]["test_files"].get("total_test_files", 0)
            total_files = self.metrics.get("code_quality", {}).get("file_statistics", {}).get("total_files", 1)
            test_ratio = (test_files / total_files) * 100 if total_files > 0 else 0
            code_quality_factors.append(min(100, test_ratio * 5))  # Test coverage proxy
        
        scores["code_quality"] = sum(code_quality_factors) / len(code_quality_factors) if code_quality_factors else 50
        
        # Security score
        security_factors = []
        
        if "security" in self.metrics:
            security_files = len(self.metrics["security"]["dependencies"].get("security_files", []))
            security_factors.append(min(100, security_files * 25))  # Security files present
            
            potential_secrets = self.metrics["security"]["secrets"].get("potential_secret_references", 0)
            security_factors.append(max(0, 100 - potential_secrets))  # Fewer potential secrets = better
            
            if "npm_audit" in self.metrics["security"]["vulnerabilities"]:
                critical_vulns = self.metrics["security"]["vulnerabilities"]["npm_audit"].get("critical", 0)
                high_vulns = self.metrics["security"]["vulnerabilities"]["npm_audit"].get("high", 0)
                vuln_penalty = (critical_vulns * 20) + (high_vulns * 10)
                security_factors.append(max(0, 100 - vuln_penalty))
        
        scores["security"] = sum(security_factors) / len(security_factors) if security_factors else 70
        
        # Maintainability score
        maintainability_factors = []
        
        if "git" in self.metrics:
            recent_commits = self.metrics["git"]["commit_activity"].get("commits_last_30_days", 0)
            maintainability_factors.append(min(100, recent_commits * 5))  # Recent activity
            
            contributors = self.metrics["git"]["repository_health"].get("contributor_count", 0)
            maintainability_factors.append(min(100, contributors * 20))  # Multiple contributors
            
            clean_working_dir = self.metrics["git"]["repository_health"].get("working_directory_clean", False)
            maintainability_factors.append(100 if clean_working_dir else 80)
        
        scores["maintainability"] = sum(maintainability_factors) / len(maintainability_factors) if maintainability_factors else 60
        
        # Performance score
        performance_factors = []
        
        if "performance" in self.metrics:
            if "npm_build" in self.metrics["performance"]["build_time"]:
                build_success = self.metrics["performance"]["build_time"]["npm_build"]["success"]
                performance_factors.append(100 if build_success else 0)
                
                build_time = self.metrics["performance"]["build_time"]["npm_build"]["duration_seconds"]
                # Penalize slow builds (> 60 seconds)
                time_score = max(0, 100 - max(0, build_time - 60))
                performance_factors.append(time_score)
        
        scores["performance"] = sum(performance_factors) / len(performance_factors) if performance_factors else 75
        
        # Overall health score (weighted average)
        weights = {
            "code_quality": 0.3,
            "security": 0.3,
            "maintainability": 0.2,
            "performance": 0.2
        }
        
        scores["overall_health"] = sum(
            scores[category] * weight for category, weight in weights.items()
        )
        
        return scores
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        logger.info("Starting comprehensive metrics collection...")
        
        self.metrics = {
            "metadata": {
                "collection_timestamp": datetime.now(timezone.utc).isoformat(),
                "repository_path": str(self.repo_path.absolute()),
                "collector_version": "1.0.0"
            }
        }
        
        # Collect different types of metrics
        metric_collectors = [
            ("git", self.collect_git_metrics),
            ("code_quality", self.collect_code_quality_metrics),
            ("test", self.collect_test_metrics),
            ("security", self.collect_security_metrics),
            ("performance", self.collect_performance_metrics),
            ("github", self.collect_github_metrics)
        ]
        
        for metric_type, collector_func in metric_collectors:
            try:
                logger.info(f"Collecting {metric_type} metrics...")
                self.metrics[metric_type] = collector_func()
            except Exception as e:
                logger.error(f"Failed to collect {metric_type} metrics: {e}")
                self.metrics[metric_type] = {"error": str(e)}
        
        # Calculate health scores
        self.metrics["health_scores"] = self.calculate_health_scores()
        
        logger.info("Metrics collection completed!")
        return self.metrics
    
    def generate_report(self, output_format: str = "json") -> str:
        """Generate a report from collected metrics."""
        if not self.metrics:
            self.collect_all_metrics()
        
        if output_format.lower() == "json":
            return json.dumps(self.metrics, indent=2, default=str)
        elif output_format.lower() == "summary":
            return self._generate_summary_report()
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        lines = []
        lines.append("=" * 60)
        lines.append("SYNTHETIC DATA GUARDIAN - METRICS REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {self.metrics['metadata']['collection_timestamp']}")
        lines.append(f"Repository: {self.metrics['metadata']['repository_path']}")
        lines.append("")
        
        # Health Scores Summary
        if "health_scores" in self.metrics:
            lines.append("📊 HEALTH SCORES")
            lines.append("-" * 30)
            for score_name, score_value in self.metrics["health_scores"].items():
                score_str = f"{score_value:.1f}/100"
                status = "🟢" if score_value >= 80 else "🟡" if score_value >= 60 else "🔴"
                lines.append(f"{status} {score_name.replace('_', ' ').title()}: {score_str}")
            lines.append("")
        
        # Repository Overview
        if "git" in self.metrics:
            lines.append("📋 REPOSITORY OVERVIEW")
            lines.append("-" * 30)
            git_metrics = self.metrics["git"]
            
            if "branch_info" in git_metrics:
                lines.append(f"Current Branch: {git_metrics['branch_info'].get('current_branch', 'N/A')}")
                
            if "commit_activity" in git_metrics:
                lines.append(f"Total Commits: {git_metrics['commit_activity'].get('total_commits', 'N/A')}")
                lines.append(f"Recent Commits (30d): {git_metrics['commit_activity'].get('commits_last_30_days', 'N/A')}")
                
            if "repository_health" in git_metrics:
                lines.append(f"Contributors: {git_metrics['repository_health'].get('contributor_count', 'N/A')}")
                clean = git_metrics['repository_health'].get('working_directory_clean', False)
                lines.append(f"Working Directory: {'Clean' if clean else 'Dirty'}")
            lines.append("")
        
        # Code Quality
        if "code_quality" in self.metrics:
            lines.append("💻 CODE QUALITY")
            lines.append("-" * 30)
            code_metrics = self.metrics["code_quality"]
            
            if "file_statistics" in code_metrics:
                lines.append(f"Total Files: {code_metrics['file_statistics'].get('total_files', 'N/A')}")
                lines.append(f"Total Lines: {code_metrics['file_statistics'].get('total_lines', 'N/A'):,}")
                avg_lines = code_metrics['file_statistics'].get('average_lines_per_file', 0)
                lines.append(f"Avg Lines/File: {avg_lines:.1f}")
                
            if "lines_of_code" in code_metrics:
                lines.append("\nLanguage Breakdown:")
                for lang, data in code_metrics["lines_of_code"].items():
                    if data["lines"] > 0:
                        lines.append(f"  {lang.title()}: {data['lines']:,} lines ({data['files']} files)")
            lines.append("")
        
        # Security Overview
        if "security" in self.metrics:
            lines.append("🔒 SECURITY")
            lines.append("-" * 30)
            security_metrics = self.metrics["security"]
            
            if "dependencies" in security_metrics:
                dep_files = security_metrics["dependencies"].get("dependency_files", [])
                lines.append(f"Dependency Files: {len(dep_files)} found")
                
                sec_files = security_metrics["dependencies"].get("security_files", [])
                lines.append(f"Security Files: {len(sec_files)} found")
                
            if "vulnerabilities" in security_metrics and "npm_audit" in security_metrics["vulnerabilities"]:
                npm_audit = security_metrics["vulnerabilities"]["npm_audit"]
                if "error" not in npm_audit:
                    total_vulns = npm_audit.get("total_vulnerabilities", 0)
                    critical = npm_audit.get("critical", 0)
                    high = npm_audit.get("high", 0)
                    lines.append(f"NPM Vulnerabilities: {total_vulns} total ({critical} critical, {high} high)")
                    
            if "secrets" in security_metrics:
                potential_secrets = security_metrics["secrets"].get("potential_secret_references", 0)
                lines.append(f"Potential Secret References: {potential_secrets}")
            lines.append("")
        
        # GitHub Statistics
        if "github" in self.metrics and "repository" in self.metrics["github"]:
            lines.append("🐙 GITHUB STATISTICS")
            lines.append("-" * 30)
            github_metrics = self.metrics["github"]
            
            if "repository" in github_metrics:
                repo = github_metrics["repository"]
                lines.append(f"Stars: {repo.get('stars', 'N/A')}")
                lines.append(f"Forks: {repo.get('forks', 'N/A')}")
                lines.append(f"Open Issues: {repo.get('open_issues', 'N/A')}")
                
            if "pull_requests" in github_metrics:
                prs = github_metrics["pull_requests"]
                lines.append(f"Open PRs: {prs.get('open', 'N/A')}")
                
            if "releases" in github_metrics:
                releases = github_metrics["releases"]
                lines.append(f"Total Releases: {releases.get('total', 'N/A')}")
                lines.append(f"Latest Release: {releases.get('latest', 'N/A')}")
            lines.append("")
        
        lines.append("=" * 60)
        lines.append("End of Report")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_report(self, filename: Optional[str] = None, format: str = "json"):
        """Save the metrics report to a file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_report_{timestamp}.{format}"
        
        report_content = self.generate_report(format)
        
        with open(filename, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to: {filename}")
        return filename


def main():
    """Main entry point for the metrics collector."""
    parser = argparse.ArgumentParser(
        description="Collect comprehensive metrics for Synthetic Data Guardian project"
    )
    parser.add_argument(
        "--repo-path", "-r",
        default=".",
        help="Path to the repository (default: current directory)"
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "summary"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--print", "-p",
        action="store_true",
        help="Print report to stdout"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create metrics collector
        collector = MetricsCollector(args.repo_path, args.config)
        
        # Collect all metrics
        metrics = collector.collect_all_metrics()
        
        # Generate and save report
        if args.output:
            collector.save_report(args.output, args.format)
        else:
            output_file = collector.save_report(format=args.format)
            print(f"Report saved to: {output_file}")
        
        # Print to stdout if requested
        if args.print:
            print("\n" + collector.generate_report(args.format))
        
        # Print summary
        overall_health = metrics.get("health_scores", {}).get("overall_health", 0)
        print(f"\n🎯 Overall Health Score: {overall_health:.1f}/100")
        
        if overall_health >= 80:
            print("✅ Excellent project health!")
        elif overall_health >= 60:
            print("⚠️  Good project health with room for improvement")
        else:
            print("🚨 Project health needs attention")
            
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()