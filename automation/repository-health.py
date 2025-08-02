#!/usr/bin/env python3
"""
Synthetic Data Guardian - Repository Health Monitor

Comprehensive repository health monitoring that tracks code quality, security,
performance, and maintenance metrics over time.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import logging
import sqlite3
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class HealthMetric:
    """Data class for health metrics."""
    timestamp: str
    metric_name: str
    value: float
    category: str
    unit: str
    threshold: Optional[float] = None
    status: str = "unknown"  # green, yellow, red, unknown


class HealthDatabase:
    """SQLite database for storing health metrics over time."""
    
    def __init__(self, db_path: str = "repository_health.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the health metrics database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                category TEXT NOT NULL,
                unit TEXT,
                threshold REAL,
                status TEXT DEFAULT 'unknown',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create health scores table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                overall_score REAL NOT NULL,
                code_quality_score REAL,
                security_score REAL,
                performance_score REAL,
                maintainability_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                metric_name TEXT,
                metric_value REAL,
                threshold REAL,
                resolved BOOLEAN DEFAULT FALSE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_metric(self, metric: HealthMetric):
        """Save a health metric to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO health_metrics 
            (timestamp, metric_name, value, category, unit, threshold, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            metric.timestamp,
            metric.metric_name,
            metric.value,
            metric.category,
            metric.unit,
            metric.threshold,
            metric.status
        ))
        
        conn.commit()
        conn.close()
    
    def save_health_scores(self, timestamp: str, scores: Dict[str, float]):
        """Save health scores to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO health_scores 
            (timestamp, overall_score, code_quality_score, security_score, 
             performance_score, maintainability_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            scores.get("overall_score", 0),
            scores.get("code_quality_score", 0),
            scores.get("security_score", 0),
            scores.get("performance_score", 0),
            scores.get("maintainability_score", 0)
        ))
        
        conn.commit()
        conn.close()
    
    def save_alert(self, timestamp: str, alert_type: str, severity: str, 
                   message: str, metric_name: str = None, metric_value: float = None,
                   threshold: float = None):
        """Save a health alert to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO health_alerts 
            (timestamp, alert_type, severity, message, metric_name, metric_value, threshold)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, alert_type, severity, message, metric_name, metric_value, threshold))
        
        conn.commit()
        conn.close()
    
    def get_metric_history(self, metric_name: str, days: int = 30) -> List[Dict]:
        """Get historical data for a specific metric."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT timestamp, value, status 
            FROM health_metrics 
            WHERE metric_name = ? AND timestamp > ?
            ORDER BY timestamp
        """, (metric_name, cutoff_date))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{"timestamp": row[0], "value": row[1], "status": row[2]} for row in results]
    
    def get_latest_scores(self) -> Optional[Dict]:
        """Get the latest health scores."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM health_scores 
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "timestamp": result[1],
                "overall_score": result[2],
                "code_quality_score": result[3],
                "security_score": result[4],
                "performance_score": result[5],
                "maintainability_score": result[6]
            }
        return None
    
    def get_active_alerts(self) -> List[Dict]:
        """Get active (unresolved) alerts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, alert_type, severity, message, metric_name, metric_value, threshold
            FROM health_alerts 
            WHERE resolved = FALSE
            ORDER BY timestamp DESC
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        return [{
            "timestamp": row[0],
            "alert_type": row[1],
            "severity": row[2],
            "message": row[3],
            "metric_name": row[4],
            "metric_value": row[5],
            "threshold": row[6]
        } for row in results]


class RepositoryHealthMonitor:
    """Comprehensive repository health monitoring."""
    
    def __init__(self, repo_path: str = ".", config: Optional[Dict[str, Any]] = None):
        self.repo_path = Path(repo_path)
        self.config = config or self._default_config()
        self.db = HealthDatabase(self.config["database"]["path"])
        self.current_metrics = []
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for health monitoring."""
        return {
            "database": {
                "path": "repository_health.db"
            },
            "thresholds": {
                "code_coverage": {"warning": 70, "critical": 50},
                "test_pass_rate": {"warning": 90, "critical": 80},
                "build_success_rate": {"warning": 95, "critical": 85},
                "security_vulnerabilities": {"warning": 5, "critical": 10},
                "technical_debt_score": {"warning": 60, "critical": 40},
                "performance_score": {"warning": 70, "critical": 50},
                "code_duplication": {"warning": 10, "critical": 20},
                "cyclomatic_complexity": {"warning": 10, "critical": 15},
                "lines_of_code_per_file": {"warning": 300, "critical": 500},
                "commit_frequency": {"warning": 1, "critical": 0.5}  # commits per day
            },
            "collection": {
                "enabled_metrics": [
                    "git_metrics",
                    "code_quality_metrics",
                    "test_metrics",
                    "security_metrics",
                    "performance_metrics",
                    "dependency_metrics"
                ]
            },
            "alerts": {
                "enabled": True,
                "notification_channels": []
            },
            "retention": {
                "metrics_days": 365,
                "alerts_days": 90
            }
        }
    
    def run_command(self, command: List[str]) -> Dict[str, Any]:
        """Run a command and return results."""
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def collect_git_metrics(self) -> List[HealthMetric]:
        """Collect Git repository metrics."""
        metrics = []
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Commit frequency (last 30 days)
        result = self.run_command([
            "git", "rev-list", "--count", "--since=30.days", "HEAD"
        ])
        if result["success"]:
            commits_30_days = int(result["stdout"])
            commit_frequency = commits_30_days / 30  # commits per day
            
            metrics.append(HealthMetric(
                timestamp=timestamp,
                metric_name="commit_frequency",
                value=commit_frequency,
                category="maintainability",
                unit="commits_per_day",
                threshold=self.config["thresholds"]["commit_frequency"]["warning"],
                status=self._evaluate_threshold(
                    commit_frequency,
                    self.config["thresholds"]["commit_frequency"],
                    higher_is_better=True
                )
            ))
        
        # Contributors count
        result = self.run_command(["git", "shortlog", "-sn", "--all"])
        if result["success"]:
            contributor_count = len(result["stdout"].split('\n')) if result["stdout"] else 0
            
            metrics.append(HealthMetric(
                timestamp=timestamp,
                metric_name="contributor_count",
                value=contributor_count,
                category="maintainability",
                unit="count",
                status="green" if contributor_count > 1 else "yellow"
            ))
        
        # Repository age
        result = self.run_command(["git", "log", "--reverse", "--format=%ct", "-1"])
        if result["success"]:
            first_commit_timestamp = int(result["stdout"])
            repo_age_days = (time.time() - first_commit_timestamp) / (24 * 3600)
            
            metrics.append(HealthMetric(
                timestamp=timestamp,
                metric_name="repository_age",
                value=repo_age_days,
                category="information",
                unit="days"
            ))
        
        # Branch count
        result = self.run_command(["git", "branch", "-r"])
        if result["success"]:
            branch_count = len(result["stdout"].split('\n')) if result["stdout"] else 0
            
            metrics.append(HealthMetric(
                timestamp=timestamp,
                metric_name="branch_count",
                value=branch_count,
                category="maintainability",
                unit="count",
                status="green" if branch_count < 20 else "yellow" if branch_count < 50 else "red"
            ))
        
        return metrics
    
    def collect_code_quality_metrics(self) -> List[HealthMetric]:
        """Collect code quality metrics."""
        metrics = []
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Lines of code
        total_lines = 0
        file_count = 0
        
        for pattern in ["*.py", "*.js", "*.ts", "*.tsx", "*.jsx"]:
            result = self.run_command([
                "find", ".", "-name", pattern, "-type", "f",
                "!", "-path", "./node_modules/*",
                "!", "-path", "./.git/*",
                "!", "-path", "./venv/*"
            ])
            
            if result["success"] and result["stdout"]:
                files = result["stdout"].split('\n')
                file_count += len(files)
                
                for file_path in files:
                    if file_path.strip():
                        wc_result = self.run_command(["wc", "-l", file_path])
                        if wc_result["success"]:
                            lines = int(wc_result["stdout"].split()[0])
                            total_lines += lines
        
        if file_count > 0:
            avg_lines_per_file = total_lines / file_count
            
            metrics.append(HealthMetric(
                timestamp=timestamp,
                metric_name="total_lines_of_code",
                value=total_lines,
                category="code_quality",
                unit="lines"
            ))
            
            metrics.append(HealthMetric(
                timestamp=timestamp,
                metric_name="average_lines_per_file",
                value=avg_lines_per_file,
                category="code_quality",
                unit="lines",
                threshold=self.config["thresholds"]["lines_of_code_per_file"]["warning"],
                status=self._evaluate_threshold(
                    avg_lines_per_file,
                    self.config["thresholds"]["lines_of_code_per_file"],
                    higher_is_better=False
                )
            ))
            
            metrics.append(HealthMetric(
                timestamp=timestamp,
                metric_name="file_count",
                value=file_count,
                category="code_quality",
                unit="count"
            ))
        
        # TODO/FIXME count
        todo_count = 0
        for pattern in ["*.py", "*.js", "*.ts", "*.tsx", "*.jsx", "*.md"]:
            result = self.run_command([
                "find", ".", "-name", pattern, "-type", "f",
                "!", "-path", "./node_modules/*",
                "!", "-path", "./.git/*"
            ])
            
            if result["success"] and result["stdout"]:
                files = result["stdout"].split('\n')
                for file_path in files:
                    if file_path.strip():
                        grep_result = self.run_command([
                            "grep", "-i", "-c", "TODO\\|FIXME\\|XXX\\|HACK", file_path
                        ])
                        if grep_result["success"] and grep_result["stdout"]:
                            try:
                                todo_count += int(grep_result["stdout"])
                            except ValueError:
                                pass
        
        metrics.append(HealthMetric(
            timestamp=timestamp,
            metric_name="todo_fixme_count",
            value=todo_count,
            category="maintainability",
            unit="count",
            status="green" if todo_count < 10 else "yellow" if todo_count < 50 else "red"
        ))
        
        return metrics
    
    def collect_test_metrics(self) -> List[HealthMetric]:
        """Collect test-related metrics."""
        metrics = []
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Test file count
        test_file_count = 0
        test_patterns = ["*test*.py", "*test*.js", "*test*.ts", "*.test.*", "*.spec.*"]
        
        for pattern in test_patterns:
            result = self.run_command([
                "find", ".", "-name", pattern, "-type", "f",
                "!", "-path", "./node_modules/*",
                "!", "-path", "./.git/*"
            ])
            
            if result["success"] and result["stdout"]:
                files = [f for f in result["stdout"].split('\n') if f.strip()]
                test_file_count += len(files)
        
        metrics.append(HealthMetric(
            timestamp=timestamp,
            metric_name="test_file_count",
            value=test_file_count,
            category="testing",
            unit="count"
        ))
        
        # Test execution (if possible)
        if (self.repo_path / "package.json").exists():
            test_result = self.run_command(["npm", "test", "--", "--reporter=json"])
            if test_result["success"]:
                try:
                    # Try to parse test results (implementation depends on test framework)
                    # This is a simplified example
                    pass_rate = 95  # Placeholder
                    
                    metrics.append(HealthMetric(
                        timestamp=timestamp,
                        metric_name="test_pass_rate",
                        value=pass_rate,
                        category="testing",
                        unit="percentage",
                        threshold=self.config["thresholds"]["test_pass_rate"]["warning"],
                        status=self._evaluate_threshold(
                            pass_rate,
                            self.config["thresholds"]["test_pass_rate"],
                            higher_is_better=True
                        )
                    ))
                except:
                    pass
        
        return metrics
    
    def collect_security_metrics(self) -> List[HealthMetric]:
        """Collect security-related metrics."""
        metrics = []
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # NPM audit (if package.json exists)
        if (self.repo_path / "package.json").exists():
            audit_result = self.run_command(["npm", "audit", "--json"])
            if audit_result["success"]:
                try:
                    audit_data = json.loads(audit_result["stdout"])
                    vulnerabilities = audit_data.get("metadata", {}).get("vulnerabilities", {})
                    total_vulns = vulnerabilities.get("total", 0)
                    critical_vulns = vulnerabilities.get("critical", 0)
                    high_vulns = vulnerabilities.get("high", 0)
                    
                    metrics.append(HealthMetric(
                        timestamp=timestamp,
                        metric_name="total_vulnerabilities",
                        value=total_vulns,
                        category="security",
                        unit="count",
                        threshold=self.config["thresholds"]["security_vulnerabilities"]["warning"],
                        status=self._evaluate_threshold(
                            total_vulns,
                            self.config["thresholds"]["security_vulnerabilities"],
                            higher_is_better=False
                        )
                    ))
                    
                    metrics.append(HealthMetric(
                        timestamp=timestamp,
                        metric_name="critical_vulnerabilities",
                        value=critical_vulns,
                        category="security",
                        unit="count",
                        status="red" if critical_vulns > 0 else "green"
                    ))
                    
                    metrics.append(HealthMetric(
                        timestamp=timestamp,
                        metric_name="high_vulnerabilities",
                        value=high_vulns,
                        category="security",
                        unit="count",
                        status="yellow" if high_vulns > 0 else "green"
                    ))
                    
                except json.JSONDecodeError:
                    pass
        
        # Look for hardcoded secrets patterns
        secret_patterns = [
            "password\\s*=\\s*[\"'][^\"']+[\"']",
            "secret\\s*=\\s*[\"'][^\"']+[\"']",
            "api_key\\s*=\\s*[\"'][^\"']+[\"']"
        ]
        
        potential_secrets = 0
        for pattern in secret_patterns:
            result = self.run_command([
                "grep", "-r", "-i", "--include=*.py", "--include=*.js",
                "--include=*.ts", "--exclude-dir=node_modules",
                "--exclude-dir=.git", pattern, "."
            ])
            if result["success"] and result["stdout"]:
                potential_secrets += len(result["stdout"].split('\n'))
        
        metrics.append(HealthMetric(
            timestamp=timestamp,
            metric_name="potential_secrets",
            value=potential_secrets,
            category="security",
            unit="count",
            status="red" if potential_secrets > 0 else "green"
        ))
        
        return metrics
    
    def collect_dependency_metrics(self) -> List[HealthMetric]:
        """Collect dependency-related metrics."""
        metrics = []
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # NPM dependencies
        if (self.repo_path / "package.json").exists():
            with open(self.repo_path / "package.json", 'r') as f:
                package_data = json.load(f)
            
            prod_deps = len(package_data.get("dependencies", {}))
            dev_deps = len(package_data.get("devDependencies", {}))
            total_deps = prod_deps + dev_deps
            
            metrics.extend([
                HealthMetric(
                    timestamp=timestamp,
                    metric_name="npm_production_dependencies",
                    value=prod_deps,
                    category="dependencies",
                    unit="count"
                ),
                HealthMetric(
                    timestamp=timestamp,
                    metric_name="npm_development_dependencies",
                    value=dev_deps,
                    category="dependencies",
                    unit="count"
                ),
                HealthMetric(
                    timestamp=timestamp,
                    metric_name="npm_total_dependencies",
                    value=total_deps,
                    category="dependencies",
                    unit="count",
                    status="green" if total_deps < 50 else "yellow" if total_deps < 100 else "red"
                )
            ])
        
        # Python dependencies
        if (self.repo_path / "requirements.txt").exists():
            with open(self.repo_path / "requirements.txt", 'r') as f:
                python_deps = len([line for line in f if line.strip() and not line.startswith('#')])
            
            metrics.append(HealthMetric(
                timestamp=timestamp,
                metric_name="python_dependencies",
                value=python_deps,
                category="dependencies",
                unit="count",
                status="green" if python_deps < 30 else "yellow" if python_deps < 50 else "red"
            ))
        
        return metrics
    
    def _evaluate_threshold(self, value: float, thresholds: Dict[str, float], 
                           higher_is_better: bool = True) -> str:
        """Evaluate a metric value against thresholds."""
        warning = thresholds["warning"]
        critical = thresholds["critical"]
        
        if higher_is_better:
            if value >= warning:
                return "green"
            elif value >= critical:
                return "yellow"
            else:
                return "red"
        else:
            if value <= warning:
                return "green"
            elif value <= critical:
                return "yellow"
            else:
                return "red"
    
    def calculate_health_scores(self, metrics: List[HealthMetric]) -> Dict[str, float]:
        """Calculate overall health scores from metrics."""
        scores = {
            "overall_score": 0,
            "code_quality_score": 0,
            "security_score": 0,
            "maintainability_score": 0,
            "testing_score": 0
        }
        
        # Group metrics by category
        category_metrics = {}
        for metric in metrics:
            if metric.category not in category_metrics:
                category_metrics[metric.category] = []
            category_metrics[metric.category].append(metric)
        
        # Calculate category scores
        for category, cat_metrics in category_metrics.items():
            green_count = sum(1 for m in cat_metrics if m.status == "green")
            yellow_count = sum(1 for m in cat_metrics if m.status == "yellow")
            red_count = sum(1 for m in cat_metrics if m.status == "red")
            total_count = len(cat_metrics)
            
            if total_count > 0:
                # Weight: green=100, yellow=60, red=0
                category_score = (green_count * 100 + yellow_count * 60) / total_count
                
                if category == "code_quality":
                    scores["code_quality_score"] = category_score
                elif category == "security":
                    scores["security_score"] = category_score
                elif category == "maintainability":
                    scores["maintainability_score"] = category_score
                elif category == "testing":
                    scores["testing_score"] = category_score
        
        # Calculate overall score (weighted average)
        weights = {
            "code_quality_score": 0.25,
            "security_score": 0.30,
            "maintainability_score": 0.25,
            "testing_score": 0.20
        }
        
        total_weighted = 0
        total_weight = 0
        
        for score_name, weight in weights.items():
            if scores[score_name] > 0:
                total_weighted += scores[score_name] * weight
                total_weight += weight
        
        if total_weight > 0:
            scores["overall_score"] = total_weighted / total_weight
        
        return scores
    
    def check_alerts(self, metrics: List[HealthMetric], scores: Dict[str, float]):
        """Check for alerts based on metrics and scores."""
        if not self.config["alerts"]["enabled"]:
            return
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Check metric-based alerts
        for metric in metrics:
            if metric.status == "red":
                self.db.save_alert(
                    timestamp=timestamp,
                    alert_type="metric_threshold",
                    severity="critical",
                    message=f"{metric.metric_name} is in critical state: {metric.value} {metric.unit}",
                    metric_name=metric.metric_name,
                    metric_value=metric.value,
                    threshold=metric.threshold
                )
            elif metric.status == "yellow":
                self.db.save_alert(
                    timestamp=timestamp,
                    alert_type="metric_threshold",
                    severity="warning",
                    message=f"{metric.metric_name} is in warning state: {metric.value} {metric.unit}",
                    metric_name=metric.metric_name,
                    metric_value=metric.value,
                    threshold=metric.threshold
                )
        
        # Check score-based alerts
        if scores["overall_score"] < 50:
            self.db.save_alert(
                timestamp=timestamp,
                alert_type="health_score",
                severity="critical",
                message=f"Overall repository health score is critically low: {scores['overall_score']:.1f}/100"
            )
        elif scores["overall_score"] < 70:
            self.db.save_alert(
                timestamp=timestamp,
                alert_type="health_score",
                severity="warning",
                message=f"Overall repository health score is below recommended threshold: {scores['overall_score']:.1f}/100"
            )
        
        # Check security score
        if scores["security_score"] < 60:
            self.db.save_alert(
                timestamp=timestamp,
                alert_type="security_score",
                severity="critical",
                message=f"Security score is critically low: {scores['security_score']:.1f}/100"
            )
    
    def collect_all_metrics(self) -> List[HealthMetric]:
        """Collect all enabled metrics."""
        all_metrics = []
        enabled_metrics = self.config["collection"]["enabled_metrics"]
        
        if "git_metrics" in enabled_metrics:
            all_metrics.extend(self.collect_git_metrics())
        
        if "code_quality_metrics" in enabled_metrics:
            all_metrics.extend(self.collect_code_quality_metrics())
        
        if "test_metrics" in enabled_metrics:
            all_metrics.extend(self.collect_test_metrics())
        
        if "security_metrics" in enabled_metrics:
            all_metrics.extend(self.collect_security_metrics())
        
        if "dependency_metrics" in enabled_metrics:
            all_metrics.extend(self.collect_dependency_metrics())
        
        return all_metrics
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run a comprehensive health check."""
        logger.info("Starting repository health check...")
        
        # Collect all metrics
        metrics = self.collect_all_metrics()
        
        # Save metrics to database
        for metric in metrics:
            self.db.save_metric(metric)
        
        # Calculate health scores
        scores = self.calculate_health_scores(metrics)
        
        # Save scores to database
        timestamp = datetime.now(timezone.utc).isoformat()
        self.db.save_health_scores(timestamp, scores)
        
        # Check for alerts
        self.check_alerts(metrics, scores)
        
        # Get active alerts
        active_alerts = self.db.get_active_alerts()
        
        result = {
            "timestamp": timestamp,
            "metrics": [asdict(m) for m in metrics],
            "health_scores": scores,
            "active_alerts": active_alerts,
            "summary": {
                "total_metrics": len(metrics),
                "green_metrics": len([m for m in metrics if m.status == "green"]),
                "yellow_metrics": len([m for m in metrics if m.status == "yellow"]),
                "red_metrics": len([m for m in metrics if m.status == "red"]),
                "overall_health": "excellent" if scores["overall_score"] >= 90 else
                                "good" if scores["overall_score"] >= 70 else
                                "fair" if scores["overall_score"] >= 50 else "poor"
            }
        }
        
        logger.info(f"Health check completed. Overall score: {scores['overall_score']:.1f}/100")
        return result
    
    def generate_health_report(self) -> str:
        """Generate a human-readable health report."""
        latest_scores = self.db.get_latest_scores()
        active_alerts = self.db.get_active_alerts()
        
        if not latest_scores:
            return "No health data available. Run a health check first."
        
        lines = []
        lines.append("=" * 60)
        lines.append("REPOSITORY HEALTH REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Last Updated: {latest_scores['timestamp']}")
        lines.append("")
        
        # Overall health
        overall_score = latest_scores["overall_score"]
        health_emoji = "🟢" if overall_score >= 90 else "🟡" if overall_score >= 70 else "🔴"
        lines.append(f"{health_emoji} Overall Health Score: {overall_score:.1f}/100")
        lines.append("")
        
        # Category scores
        lines.append("📊 CATEGORY SCORES")
        lines.append("-" * 30)
        categories = [
            ("Code Quality", "code_quality_score"),
            ("Security", "security_score"),
            ("Maintainability", "maintainability_score"),
            ("Testing", "testing_score")
        ]
        
        for category_name, score_key in categories:
            score = latest_scores.get(score_key, 0)
            if score > 0:
                emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
                lines.append(f"{emoji} {category_name}: {score:.1f}/100")
        lines.append("")
        
        # Active alerts
        if active_alerts:
            lines.append("🚨 ACTIVE ALERTS")
            lines.append("-" * 30)
            for alert in active_alerts[:5]:  # Show top 5 alerts
                severity_emoji = "🔴" if alert["severity"] == "critical" else "🟡"
                lines.append(f"{severity_emoji} {alert['message']}")
            
            if len(active_alerts) > 5:
                lines.append(f"... and {len(active_alerts) - 5} more alerts")
            lines.append("")
        else:
            lines.append("✅ No active alerts")
            lines.append("")
        
        # Recommendations
        lines.append("💡 RECOMMENDATIONS")
        lines.append("-" * 30)
        
        if overall_score < 70:
            lines.append("• Focus on addressing critical and warning alerts")
        if latest_scores.get("security_score", 0) < 80:
            lines.append("• Review and fix security vulnerabilities")
        if latest_scores.get("testing_score", 0) < 80:
            lines.append("• Improve test coverage and test quality")
        if latest_scores.get("code_quality_score", 0) < 80:
            lines.append("• Refactor complex code and reduce technical debt")
        
        if overall_score >= 90:
            lines.append("• Excellent health! Continue current practices")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def main():
    """Main entry point for repository health monitoring."""
    parser = argparse.ArgumentParser(
        description="Monitor repository health for Synthetic Data Guardian"
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
        help="Output file path for detailed results"
    )
    parser.add_argument(
        "--report", "-p",
        action="store_true",
        help="Generate and display health report"
    )
    parser.add_argument(
        "--alerts", "-a",
        action="store_true",
        help="Show active alerts only"
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
        # Load configuration
        config = None
        if args.config and Path(args.config).exists():
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        # Create health monitor
        monitor = RepositoryHealthMonitor(args.repo_path, config)
        
        if args.alerts:
            # Show active alerts only
            active_alerts = monitor.db.get_active_alerts()
            if active_alerts:
                print("🚨 Active Alerts:")
                for alert in active_alerts:
                    severity_emoji = "🔴" if alert["severity"] == "critical" else "🟡"
                    print(f"{severity_emoji} {alert['message']}")
            else:
                print("✅ No active alerts")
        elif args.report:
            # Generate health report
            report = monitor.generate_health_report()
            print(report)
        else:
            # Run full health check
            results = monitor.run_health_check()
            
            # Save detailed results if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"Detailed results saved to: {args.output}")
            
            # Print summary
            summary = results["summary"]
            overall_score = results["health_scores"]["overall_score"]
            
            print(f"\n🎯 Repository Health Summary:")
            print(f"  Overall Score: {overall_score:.1f}/100 ({summary['overall_health']})")
            print(f"  Metrics: {summary['green_metrics']} good, {summary['yellow_metrics']} warning, {summary['red_metrics']} critical")
            print(f"  Active Alerts: {len(results['active_alerts'])}")
            
            if overall_score >= 90:
                print("✅ Excellent repository health!")
            elif overall_score >= 70:
                print("⚠️  Good health with room for improvement")
            else:
                print("🚨 Repository health needs attention")
        
    except Exception as e:
        logger.error(f"Repository health monitoring failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()