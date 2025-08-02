#!/usr/bin/env python3
"""
Synthetic Data Guardian - Technical Debt Tracker

Automated detection and tracking of technical debt across the codebase.
Identifies code smells, TODO comments, deprecated patterns, and maintenance issues.
"""

import ast
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TechnicalDebtTracker:
    """Tracks and analyzes technical debt in the codebase."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.debt_items = []
        self.patterns = self._load_debt_patterns()
        
    def _load_debt_patterns(self) -> Dict[str, Any]:
        """Load patterns that indicate technical debt."""
        return {
            "todo_patterns": [
                r"TODO[:|\s]",
                r"FIXME[:|\s]",
                r"HACK[:|\s]",
                r"XXX[:|\s]",
                r"BUG[:|\s]",
                r"TEMP[:|\s]",
                r"TEMPORARY[:|\s]",
                r"KLUDGE[:|\s]",
                r"WORKAROUND[:|\s]"
            ],
            "code_smells": {
                "python": [
                    r"except\s*:",  # Bare except
                    r"eval\s*\(",  # Use of eval
                    r"exec\s*\(",  # Use of exec
                    r"import\s+\*",  # Star imports
                    r"global\s+\w+",  # Global variables
                    r"def\s+\w+\s*\([^)]{100,}\)",  # Long parameter lists
                ],
                "javascript": [
                    r"console\.log\s*\(",  # Console.log statements
                    r"debugger[;\s]",  # Debugger statements
                    r"eval\s*\(",  # Use of eval
                    r"with\s*\(",  # Use of with statement
                    r"var\s+\w+",  # Use of var instead of let/const
                ],
                "typescript": [
                    r"any\s*[;\|\&\>\<\)]",  # Use of any type
                    r"@ts-ignore",  # TypeScript ignore comments
                    r"@ts-expect-error",  # TypeScript expect error
                    r"console\.log\s*\(",  # Console.log statements
                ]
            },
            "deprecated_patterns": {
                "python": [
                    r"imp\.load_source",  # Deprecated import method
                    r"platform\.dist",  # Deprecated platform method
                    r"asyncio\.coroutine",  # Deprecated decorator
                ],
                "javascript": [
                    r"\.substr\s*\(",  # Deprecated substr
                    r"new Buffer\s*\(",  # Deprecated Buffer constructor
                ],
                "general": [
                    r"http://",  # Insecure HTTP
                ]
            },
            "complexity_indicators": [
                r"if\s+.*\s+and\s+.*\s+and\s+.*\s+and",  # Complex conditions
                r"elif.*elif.*elif",  # Long elif chains
                r"for\s+.*\s+in\s+.*\s+for\s+.*\s+in",  # Nested loops in comprehensions
            ],
            "security_concerns": [
                r"password\s*=\s*[\"'][^\"']*[\"']",  # Hardcoded passwords
                r"secret\s*=\s*[\"'][^\"']*[\"']",  # Hardcoded secrets
                r"token\s*=\s*[\"'][^\"']*[\"']",  # Hardcoded tokens
                r"api_key\s*=\s*[\"'][^\"']*[\"']",  # Hardcoded API keys
                r"subprocess\.call\s*\(",  # Potentially unsafe subprocess
                r"os\.system\s*\(",  # Potentially unsafe system calls
            ]
        }
    
    def find_files_to_analyze(self) -> List[Path]:
        """Find all files that should be analyzed for technical debt."""
        file_patterns = [
            "**/*.py",
            "**/*.js",
            "**/*.ts",
            "**/*.tsx",
            "**/*.jsx",
            "**/*.yml",
            "**/*.yaml",
            "**/*.json",
            "**/*.md",
            "**/*.sh",
            "**/Dockerfile*",
            "**/Makefile*"
        ]
        
        exclude_patterns = [
            "**/node_modules/**",
            "**/.git/**",
            "**/venv/**",
            "**/__pycache__/**",
            "**/dist/**",
            "**/build/**",
            "**/.pytest_cache/**",
            "**/coverage/**",
            "**/*.min.js",
            "**/*.min.css"
        ]
        
        files = []
        for pattern in file_patterns:
            for file_path in self.repo_path.glob(pattern):
                if file_path.is_file():
                    # Check if file should be excluded
                    exclude_file = False
                    for exclude_pattern in exclude_patterns:
                        if file_path.match(exclude_pattern):
                            exclude_file = True
                            break
                    
                    if not exclude_file:
                        files.append(file_path)
        
        return files
    
    def analyze_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze a single file for technical debt."""
        debt_items = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Determine file type
            file_type = self._get_file_type(file_path)
            
            # Analyze line by line
            for line_num, line in enumerate(lines, 1):
                line_debt = self._analyze_line(line, line_num, file_path, file_type)
                debt_items.extend(line_debt)
            
            # Analyze file structure for Python files
            if file_type == "python":
                structure_debt = self._analyze_python_structure(content, file_path)
                debt_items.extend(structure_debt)
            
            # Check file-level issues
            file_debt = self._analyze_file_level(file_path, content, lines)
            debt_items.extend(file_debt)
            
        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
            debt_items.append({
                "type": "analysis_error",
                "file": str(file_path),
                "line": 0,
                "description": f"Failed to analyze file: {e}",
                "severity": "low",
                "category": "tooling"
            })
        
        return debt_items
    
    def _get_file_type(self, file_path: Path) -> str:
        """Determine the file type based on extension."""
        extension = file_path.suffix.lower()
        name = file_path.name.lower()
        
        if extension == ".py":
            return "python"
        elif extension in [".js", ".jsx"]:
            return "javascript"
        elif extension in [".ts", ".tsx"]:
            return "typescript"
        elif extension in [".yml", ".yaml"]:
            return "yaml"
        elif extension == ".json":
            return "json"
        elif extension == ".md":
            return "markdown"
        elif extension == ".sh":
            return "shell"
        elif "dockerfile" in name:
            return "dockerfile"
        elif "makefile" in name:
            return "makefile"
        else:
            return "general"
    
    def _analyze_line(self, line: str, line_num: int, file_path: Path, file_type: str) -> List[Dict[str, Any]]:
        """Analyze a single line for technical debt patterns."""
        debt_items = []
        line_stripped = line.strip()
        
        # Skip empty lines and pure comments
        if not line_stripped or (line_stripped.startswith('#') and len(line_stripped) < 5):
            return debt_items
        
        # Check for TODO patterns
        for pattern in self.patterns["todo_patterns"]:
            if re.search(pattern, line, re.IGNORECASE):
                debt_items.append({
                    "type": "todo_comment",
                    "file": str(file_path),
                    "line": line_num,
                    "description": f"TODO/FIXME comment: {line.strip()}",
                    "severity": "medium",
                    "category": "maintenance",
                    "content": line.strip()
                })
        
        # Check for code smells specific to file type
        if file_type in self.patterns["code_smells"]:
            for pattern in self.patterns["code_smells"][file_type]:
                if re.search(pattern, line):
                    debt_items.append({
                        "type": "code_smell",
                        "file": str(file_path),
                        "line": line_num,
                        "description": f"Code smell detected: {pattern}",
                        "severity": "medium",
                        "category": "code_quality",
                        "pattern": pattern,
                        "content": line.strip()
                    })
        
        # Check for deprecated patterns
        if file_type in self.patterns["deprecated_patterns"]:
            for pattern in self.patterns["deprecated_patterns"][file_type]:
                if re.search(pattern, line):
                    debt_items.append({
                        "type": "deprecated_code",
                        "file": str(file_path),
                        "line": line_num,
                        "description": f"Deprecated pattern: {pattern}",
                        "severity": "high",
                        "category": "modernization",
                        "pattern": pattern,
                        "content": line.strip()
                    })
        
        # Check general deprecated patterns
        for pattern in self.patterns["deprecated_patterns"]["general"]:
            if re.search(pattern, line):
                debt_items.append({
                    "type": "deprecated_code",
                    "file": str(file_path),
                    "line": line_num,
                    "description": f"Deprecated pattern: {pattern}",
                    "severity": "high",
                    "category": "security",
                    "pattern": pattern,
                    "content": line.strip()
                })
        
        # Check for complexity indicators
        for pattern in self.patterns["complexity_indicators"]:
            if re.search(pattern, line):
                debt_items.append({
                    "type": "complexity",
                    "file": str(file_path),
                    "line": line_num,
                    "description": f"Complex code pattern: {pattern}",
                    "severity": "medium",
                    "category": "complexity",
                    "pattern": pattern,
                    "content": line.strip()
                })
        
        # Check for security concerns
        for pattern in self.patterns["security_concerns"]:
            if re.search(pattern, line):
                debt_items.append({
                    "type": "security_concern",
                    "file": str(file_path),
                    "line": line_num,
                    "description": f"Potential security issue: {pattern}",
                    "severity": "high",
                    "category": "security",
                    "pattern": pattern,
                    "content": line.strip()
                })
        
        # Check for long lines
        if len(line) > 120:
            debt_items.append({
                "type": "long_line",
                "file": str(file_path),
                "line": line_num,
                "description": f"Long line ({len(line)} characters)",
                "severity": "low",
                "category": "code_style",
                "length": len(line)
            })
        
        return debt_items
    
    def _analyze_python_structure(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze Python file structure for debt indicators."""
        debt_items = []
        
        try:
            tree = ast.parse(content)
            
            # Check for complex functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Count complexity indicators
                    complexity_score = self._calculate_function_complexity(node)
                    
                    if complexity_score > 10:
                        debt_items.append({
                            "type": "complex_function",
                            "file": str(file_path),
                            "line": node.lineno,
                            "description": f"Complex function '{node.name}' (complexity: {complexity_score})",
                            "severity": "medium",
                            "category": "complexity",
                            "function_name": node.name,
                            "complexity_score": complexity_score
                        })
                    
                    # Check for long parameter lists
                    param_count = len(node.args.args)
                    if param_count > 5:
                        debt_items.append({
                            "type": "long_parameter_list",
                            "file": str(file_path),
                            "line": node.lineno,
                            "description": f"Function '{node.name}' has {param_count} parameters",
                            "severity": "medium",
                            "category": "code_design",
                            "function_name": node.name,
                            "parameter_count": param_count
                        })
                
                # Check for large classes
                elif isinstance(node, ast.ClassDef):
                    method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                    
                    if method_count > 15:
                        debt_items.append({
                            "type": "large_class",
                            "file": str(file_path),
                            "line": node.lineno,
                            "description": f"Large class '{node.name}' with {method_count} methods",
                            "severity": "medium",
                            "category": "code_design",
                            "class_name": node.name,
                            "method_count": method_count
                        })
                        
        except SyntaxError as e:
            debt_items.append({
                "type": "syntax_error",
                "file": str(file_path),
                "line": e.lineno or 0,
                "description": f"Syntax error: {e.msg}",
                "severity": "high",
                "category": "correctness"
            })
        except Exception as e:
            logger.warning(f"Failed to parse Python file {file_path}: {e}")
        
        return debt_items
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _analyze_file_level(self, file_path: Path, content: str, lines: List[str]) -> List[Dict[str, Any]]:
        """Analyze file-level technical debt indicators."""
        debt_items = []
        
        # Check file size
        line_count = len(lines)
        if line_count > 500:
            debt_items.append({
                "type": "large_file",
                "file": str(file_path),
                "line": 0,
                "description": f"Large file with {line_count} lines",
                "severity": "medium",
                "category": "code_organization",
                "line_count": line_count
            })
        
        # Check for missing docstrings in Python files
        if file_path.suffix == ".py" and line_count > 10:
            has_module_docstring = False
            for line in lines[:10]:
                if '"""' in line or "'''" in line:
                    has_module_docstring = True
                    break
            
            if not has_module_docstring:
                debt_items.append({
                    "type": "missing_docstring",
                    "file": str(file_path),
                    "line": 1,
                    "description": "Missing module docstring",
                    "severity": "low",
                    "category": "documentation"
                })
        
        # Check for files without tests
        if file_path.suffix in [".py", ".js", ".ts"] and "test" not in str(file_path).lower():
            # Look for corresponding test file
            test_patterns = [
                f"test_{file_path.stem}.py",
                f"{file_path.stem}_test.py",
                f"{file_path.stem}.test.js",
                f"{file_path.stem}.test.ts",
                f"{file_path.stem}.spec.js",
                f"{file_path.stem}.spec.ts"
            ]
            
            has_test = False
            for pattern in test_patterns:
                test_paths = list(self.repo_path.rglob(pattern))
                if test_paths:
                    has_test = True
                    break
            
            if not has_test:
                debt_items.append({
                    "type": "missing_tests",
                    "file": str(file_path),
                    "line": 0,
                    "description": "No corresponding test file found",
                    "severity": "medium",
                    "category": "testing"
                })
        
        return debt_items
    
    def analyze_repository(self) -> Dict[str, Any]:
        """Analyze the entire repository for technical debt."""
        logger.info("Starting technical debt analysis...")
        
        files_to_analyze = self.find_files_to_analyze()
        logger.info(f"Found {len(files_to_analyze)} files to analyze")
        
        all_debt_items = []
        file_count = 0
        
        for file_path in files_to_analyze:
            file_count += 1
            if file_count % 50 == 0:
                logger.info(f"Analyzed {file_count}/{len(files_to_analyze)} files...")
            
            debt_items = self.analyze_file(file_path)
            all_debt_items.extend(debt_items)
        
        self.debt_items = all_debt_items
        logger.info(f"Analysis complete. Found {len(all_debt_items)} debt items.")
        
        return self.generate_summary()
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of technical debt findings."""
        summary = {
            "metadata": {
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "repository_path": str(self.repo_path.absolute()),
                "total_debt_items": len(self.debt_items)
            },
            "debt_items": self.debt_items,
            "summary_stats": {},
            "recommendations": []
        }
        
        # Calculate summary statistics
        stats = {
            "by_severity": defaultdict(int),
            "by_category": defaultdict(int),
            "by_type": defaultdict(int),
            "by_file": defaultdict(int),
            "total_files_with_debt": len(set(item["file"] for item in self.debt_items))
        }
        
        for item in self.debt_items:
            stats["by_severity"][item["severity"]] += 1
            stats["by_category"][item["category"]] += 1
            stats["by_type"][item["type"]] += 1
            stats["by_file"][item["file"]] += 1
        
        summary["summary_stats"] = dict(stats)
        
        # Generate recommendations
        recommendations = []
        
        high_severity_count = stats["by_severity"]["high"]
        if high_severity_count > 0:
            recommendations.append({
                "priority": "high",
                "description": f"Address {high_severity_count} high-severity debt items immediately",
                "category": "urgent"
            })
        
        todo_count = stats["by_type"]["todo_comment"]
        if todo_count > 10:
            recommendations.append({
                "priority": "medium",
                "description": f"Review and resolve {todo_count} TODO/FIXME comments",
                "category": "maintenance"
            })
        
        complex_functions = stats["by_type"]["complex_function"]
        if complex_functions > 0:
            recommendations.append({
                "priority": "medium",
                "description": f"Refactor {complex_functions} complex functions to improve maintainability",
                "category": "refactoring"
            })
        
        missing_tests = stats["by_type"]["missing_tests"]
        if missing_tests > 0:
            recommendations.append({
                "priority": "medium",
                "description": f"Add tests for {missing_tests} files without test coverage",
                "category": "testing"
            })
        
        security_concerns = stats["by_category"]["security"]
        if security_concerns > 0:
            recommendations.append({
                "priority": "high",
                "description": f"Review and fix {security_concerns} potential security issues",
                "category": "security"
            })
        
        summary["recommendations"] = recommendations
        
        # Calculate debt score (0-100, where 100 is no debt)
        total_files = len(set(item["file"] for item in self.debt_items)) or 1
        debt_density = len(self.debt_items) / total_files
        
        # Weighted penalty by severity
        severity_weights = {"high": 10, "medium": 5, "low": 1}
        weighted_debt = sum(severity_weights.get(item["severity"], 1) for item in self.debt_items)
        
        # Calculate score (inverse of debt, normalized)
        max_penalty = total_files * 100  # Arbitrary maximum
        debt_score = max(0, 100 - min(100, (weighted_debt / max_penalty) * 100))
        
        summary["debt_score"] = round(debt_score, 1)
        
        return summary
    
    def get_top_debt_files(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get files with the most technical debt items."""
        file_debt_count = defaultdict(int)
        
        for item in self.debt_items:
            file_debt_count[item["file"]] += 1
        
        return sorted(file_debt_count.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def get_debt_by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        """Group debt items by category."""
        categorized_debt = defaultdict(list)
        
        for item in self.debt_items:
            categorized_debt[item["category"]].append(item)
        
        return dict(categorized_debt)
    
    def save_report(self, filename: Optional[str] = None, format: str = "json"):
        """Save the technical debt report to a file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tech_debt_report_{timestamp}.{format}"
        
        summary = self.generate_summary()
        
        if format.lower() == "json":
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        elif format.lower() == "csv":
            self._save_csv_report(filename, summary)
        elif format.lower() == "html":
            self._save_html_report(filename, summary)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Technical debt report saved to: {filename}")
        return filename
    
    def _save_csv_report(self, filename: str, summary: Dict[str, Any]):
        """Save debt items as CSV."""
        import csv
        
        with open(filename, 'w', newline='') as f:
            if summary["debt_items"]:
                fieldnames = summary["debt_items"][0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary["debt_items"])
    
    def _save_html_report(self, filename: str, summary: Dict[str, Any]):
        """Save debt report as HTML."""
        html_content = self._generate_html_report(summary)
        with open(filename, 'w') as f:
            f.write(html_content)
    
    def _generate_html_report(self, summary: Dict[str, Any]) -> str:
        """Generate HTML report content."""
        debt_score = summary.get("debt_score", 0)
        total_items = summary["metadata"]["total_debt_items"]
        stats = summary["summary_stats"]
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Technical Debt Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .score {{ font-size: 24px; font-weight: bold; }}
        .good {{ color: green; }}
        .warning {{ color: orange; }}
        .danger {{ color: red; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat-box {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; flex: 1; }}
        .debt-item {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
        .high {{ border-left-color: #dc3545; }}
        .medium {{ border-left-color: #ffc107; }}
        .low {{ border-left-color: #28a745; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Technical Debt Report</h1>
        <div class="score {'good' if debt_score >= 80 else 'warning' if debt_score >= 60 else 'danger'}">
            Debt Score: {debt_score}/100
        </div>
        <p>Generated: {summary['metadata']['analysis_timestamp']}</p>
        <p>Total Debt Items: {total_items}</p>
    </div>
    
    <div class="stats">
        <div class="stat-box">
            <h3>By Severity</h3>
            <ul>
                <li>High: {stats['by_severity'].get('high', 0)}</li>
                <li>Medium: {stats['by_severity'].get('medium', 0)}</li>
                <li>Low: {stats['by_severity'].get('low', 0)}</li>
            </ul>
        </div>
        <div class="stat-box">
            <h3>By Category</h3>
            <ul>
        """
        
        for category, count in stats["by_category"].items():
            html += f"<li>{category.title()}: {count}</li>"
        
        html += """
            </ul>
        </div>
    </div>
    
    <h2>Recommendations</h2>
    <ul>
        """
        
        for rec in summary["recommendations"]:
            html += f"<li><strong>[{rec['priority'].upper()}]</strong> {rec['description']}</li>"
        
        html += """
    </ul>
    
    <h2>Debt Items</h2>
        """
        
        for item in summary["debt_items"][:50]:  # Limit to first 50 items
            severity_class = item["severity"]
            html += f"""
    <div class="debt-item {severity_class}">
        <strong>{item['type'].replace('_', ' ').title()}</strong> 
        [{item['severity'].upper()}] in {item['file']}:{item['line']}<br>
        {item['description']}
    </div>
            """
        
        if len(summary["debt_items"]) > 50:
            html += f"<p><em>... and {len(summary['debt_items']) - 50} more items</em></p>"
        
        html += """
</body>
</html>
        """
        
        return html


def main():
    """Main entry point for the technical debt tracker."""
    parser = argparse.ArgumentParser(
        description="Track technical debt in Synthetic Data Guardian project"
    )
    parser.add_argument(
        "--repo-path", "-r",
        default=".",
        help="Path to the repository (default: current directory)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "csv", "html"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--summary-only", "-s",
        action="store_true",
        help="Show only summary statistics"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create technical debt tracker
        tracker = TechnicalDebtTracker(args.repo_path)
        
        # Analyze repository
        summary = tracker.analyze_repository()
        
        # Save report
        if args.output:
            tracker.save_report(args.output, args.format)
        else:
            output_file = tracker.save_report(format=args.format)
            print(f"Technical debt report saved to: {output_file}")
        
        # Print summary
        debt_score = summary.get("debt_score", 0)
        total_items = summary["metadata"]["total_debt_items"]
        
        print(f"\n🎯 Technical Debt Score: {debt_score}/100")
        print(f"📊 Total Debt Items: {total_items}")
        
        if args.summary_only:
            print("\n📈 Summary by Severity:")
            for severity, count in summary["summary_stats"]["by_severity"].items():
                print(f"  {severity.upper()}: {count}")
            
            print("\n📂 Summary by Category:")
            for category, count in summary["summary_stats"]["by_category"].items():
                print(f"  {category.title()}: {count}")
            
            print("\n🔥 Top 5 Files with Most Debt:")
            top_files = tracker.get_top_debt_files(5)
            for file_path, debt_count in top_files:
                print(f"  {debt_count} items: {file_path}")
        
        # Status message
        if debt_score >= 80:
            print("✅ Low technical debt - well maintained codebase!")
        elif debt_score >= 60:
            print("⚠️  Moderate technical debt - consider addressing high-priority items")
        else:
            print("🚨 High technical debt - immediate attention recommended")
            
    except Exception as e:
        logger.error(f"Technical debt analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()