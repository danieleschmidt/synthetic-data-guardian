#!/usr/bin/env node

/**
 * Terragon Autonomous Value Discovery Engine
 * 
 * Implements WSJF + ICE + Technical Debt scoring for continuous value delivery
 * Discovers and prioritizes work items from multiple sources
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import yaml from 'js-yaml';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class ValueDiscoveryEngine {
  constructor() {
    this.config = this.loadConfig();
    this.workItems = [];
    this.metrics = {
      totalDiscovered: 0,
      totalCompleted: 0,
      averageScore: 0,
      executionHistory: []
    };
  }

  loadConfig() {
    try {
      const configPath = path.join(__dirname, 'value-config.yaml');
      const configFile = fs.readFileSync(configPath, 'utf8');
      return yaml.load(configFile);
    } catch (error) {
      console.error('Failed to load configuration:', error.message);
      return this.getDefaultConfig();
    }
  }

  getDefaultConfig() {
    return {
      scoring: {
        weights: {
          maturing: { wsjf: 0.5, ice: 0.2, technicalDebt: 0.2, security: 0.1 }
        },
        thresholds: { minScore: 15.0, maxRisk: 0.8, securityBoost: 2.0 }
      }
    };
  }

  /**
   * Main discovery method - harvests signals from all sources
   */
  async discoverWorkItems() {
    console.log('🔍 Starting comprehensive value discovery...');
    
    const sources = [
      this.discoverFromGitHistory(),
      this.discoverFromCodeComments(),
      this.discoverFromStaticAnalysis(),
      this.discoverFromDependencies(),
      this.discoverFromPerformance(),
      this.discoverFromSecurity(),
      this.discoverFromTestCoverage(),
      this.discoverFromDocumentation()
    ];

    const allItems = await Promise.all(sources);
    this.workItems = allItems.flat();
    
    console.log(`📊 Discovered ${this.workItems.length} potential work items`);
    return this.workItems;
  }

  /**
   * Git history analysis for patterns and debt indicators
   */
  discoverFromGitHistory() {
    const items = [];
    
    try {
      // Look for quick fixes and temporary solutions
      const quickFixes = execSync('git log --since="3 months ago" --grep="quick\\|temp\\|hack\\|fix" --oneline --no-merges', { encoding: 'utf8' });
      
      quickFixes.split('\n').filter(line => line.trim()).forEach(commit => {
        const match = commit.match(/^([a-f0-9]+)\s+(.+)$/);
        if (match) {
          items.push({
            id: `git-${match[1].substring(0, 7)}`,
            title: `Review quick fix: ${match[2]}`,
            type: 'technical-debt',
            source: 'git-history',
            description: `Quick fix commit may need proper solution: ${match[2]}`,
            effort: 4,
            impact: 'medium',
            confidence: 0.7,
            files: [],
            metadata: { commit: match[1], message: match[2] }
          });
        }
      });

      // Analyze churn patterns
      const highChurnFiles = execSync('git log --since="1 month ago" --name-only --pretty=format: | sort | uniq -c | sort -nr | head -10', { encoding: 'utf8' });
      
      highChurnFiles.split('\n').filter(line => line.trim() && !line.includes('node_modules')).forEach(line => {
        const match = line.trim().match(/^\s*(\d+)\s+(.+)$/);
        if (match && parseInt(match[1]) > 5) {
          items.push({
            id: `churn-${Buffer.from(match[2]).toString('base64').substring(0, 8)}`,
            title: `Refactor high-churn file: ${path.basename(match[2])}`,
            type: 'technical-debt',
            source: 'git-churn',
            description: `File changed ${match[1]} times recently, may need refactoring`,
            effort: 6,
            impact: 'high',
            confidence: 0.8,
            files: [match[2]],
            metadata: { churnCount: parseInt(match[1]), file: match[2] }
          });
        }
      });

    } catch (error) {
      console.warn('Git history analysis failed:', error.message);
    }
    
    return items;
  }

  /**
   * Parse code comments for TODO, FIXME, HACK, etc.
   */
  discoverFromCodeComments() {
    const items = [];
    
    try {
      // Search for debt markers in code
      const debtMarkers = execSync('find . -type f \\( -name "*.js" -o -name "*.ts" -o -name "*.py" \\) -not -path "./node_modules/*" -not -path "./.git/*" | xargs grep -Hn "TODO\\|FIXME\\|HACK\\|BUG\\|DEPRECATED\\|XXX" | head -20', { encoding: 'utf8' });
      
      debtMarkers.split('\n').filter(line => line.trim()).forEach(line => {
        const match = line.match(/^([^:]+):(\d+):(.+)$/);
        if (match) {
          const [, file, lineNum, comment] = match;
          const marker = comment.match(/(TODO|FIXME|HACK|BUG|DEPRECATED|XXX)/i);
          
          if (marker) {
            items.push({
              id: `comment-${Buffer.from(`${file}:${lineNum}`).toString('base64').substring(0, 8)}`,
              title: `Address ${marker[1]}: ${path.basename(file)}:${lineNum}`,
              type: 'technical-debt',
              source: 'code-comments',
              description: comment.trim(),
              effort: this.estimateEffortFromComment(comment),
              impact: this.assessImpactFromMarker(marker[1]),
              confidence: 0.9,
              files: [file],
              metadata: { file, line: parseInt(lineNum), marker: marker[1], comment }
            });
          }
        }
      });
    } catch (error) {
      console.warn('Code comment analysis failed:', error.message);
    }
    
    return items;
  }

  /**
   * Static analysis using available tools
   */
  discoverFromStaticAnalysis() {
    const items = [];
    
    try {
      // TypeScript compiler issues
      const tscIssues = execSync('npx tsc --noEmit --listFiles 2>&1 | grep "error TS" | head -10 || true', { encoding: 'utf8' });
      
      tscIssues.split('\n').filter(line => line.includes('error TS')).forEach(issue => {
        const match = issue.match(/([^(]+)\((\d+),(\d+)\): error TS(\d+): (.+)/);
        if (match) {
          const [, file, line, col, code, message] = match;
          items.push({
            id: `ts-${code}-${Buffer.from(`${file}:${line}`).toString('base64').substring(0, 6)}`,
            title: `Fix TypeScript error: ${path.basename(file)}:${line}`,
            type: 'bug-fix',
            source: 'typescript',
            description: `TS${code}: ${message}`,
            effort: 2,
            impact: 'medium',
            confidence: 0.95,
            files: [file],
            metadata: { file, line: parseInt(line), code, message }
          });
        }
      });

      // ESLint issues (if available)
      try {
        const eslintIssues = execSync('npx eslint src/ --format json --max-warnings 0 2>/dev/null || echo "[]"', { encoding: 'utf8' });
        const lintResults = JSON.parse(eslintIssues);
        
        lintResults.forEach(result => {
          result.messages?.forEach(message => {
            if (message.severity === 2) { // Errors only
              items.push({
                id: `eslint-${message.ruleId}-${Buffer.from(`${result.filePath}:${message.line}`).toString('base64').substring(0, 6)}`,
                title: `Fix ESLint error: ${path.basename(result.filePath)}:${message.line}`,
                type: 'code-quality',
                source: 'eslint',
                description: `${message.message} (${message.ruleId})`,
                effort: 1,
                impact: 'low',
                confidence: 0.8,
                files: [result.filePath],
                metadata: { rule: message.ruleId, line: message.line, message: message.message }
              });
            }
          });
        });
      } catch (eslintError) {
        console.warn('ESLint analysis skipped:', eslintError.message);
      }

    } catch (error) {
      console.warn('Static analysis failed:', error.message);
    }
    
    return items;
  }

  /**
   * Dependency vulnerability and update analysis
   */
  discoverFromDependencies() {
    const items = [];
    
    try {
      // Check for outdated dependencies
      const outdated = execSync('npm outdated --json 2>/dev/null || echo "{}"', { encoding: 'utf8' });
      const outdatedDeps = JSON.parse(outdated);
      
      Object.entries(outdatedDeps).forEach(([name, info]) => {
        const isSecurityUpdate = this.isSecurityUpdate(info);
        items.push({
          id: `dep-${name}`,
          title: `Update ${name} from ${info.current} to ${info.latest}`,
          type: isSecurityUpdate ? 'security-update' : 'dependency-update',
          source: 'npm-outdated',
          description: `Update ${name}: ${info.current} → ${info.latest}`,
          effort: 2,
          impact: isSecurityUpdate ? 'high' : 'low',
          confidence: 0.9,
          files: ['package.json'],
          metadata: { package: name, current: info.current, latest: info.latest, wanted: info.wanted }
        });
      });

      // Security audit
      const auditResult = execSync('npm audit --json 2>/dev/null || echo "{\\"vulnerabilities\\":{}}"', { encoding: 'utf8' });
      const audit = JSON.parse(auditResult);
      
      Object.entries(audit.vulnerabilities || {}).forEach(([name, vuln]) => {
        items.push({
          id: `vuln-${name}`,
          title: `Fix ${vuln.severity} vulnerability in ${name}`,
          type: 'security-fix',
          source: 'npm-audit',
          description: `${vuln.severity} severity vulnerability: ${vuln.title}`,
          effort: this.estimateSecurityFixEffort(vuln.severity),
          impact: 'high',
          confidence: 0.95,
          files: ['package.json'],
          metadata: { package: name, severity: vuln.severity, title: vuln.title }
        });
      });

    } catch (error) {
      console.warn('Dependency analysis failed:', error.message);
    }
    
    return items;
  }

  /**
   * Performance analysis and optimization opportunities
   */
  discoverFromPerformance() {
    const items = [];
    
    // Analyze bundle size (if applicable)
    try {
      const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
      
      if (packageJson.dependencies) {
        const largeDeps = Object.keys(packageJson.dependencies).filter(dep => 
          ['lodash', 'moment', 'webpack', 'babel'].some(large => dep.includes(large))
        );
        
        largeDeps.forEach(dep => {
          items.push({
            id: `perf-${dep}`,
            title: `Optimize large dependency: ${dep}`,
            type: 'performance',
            source: 'bundle-analysis',
            description: `Consider tree-shaking or alternatives for ${dep}`,
            effort: 4,
            impact: 'medium',
            confidence: 0.6,
            files: ['package.json'],
            metadata: { dependency: dep, type: 'bundle-size' }
          });
        });
      }
      
      // Check for performance test failures or regressions
      if (fs.existsSync('tests/performance')) {
        items.push({
          id: 'perf-monitoring',
          title: 'Enhance performance monitoring',
          type: 'performance',
          source: 'performance-tests',
          description: 'Add more comprehensive performance benchmarks',
          effort: 6,
          impact: 'medium',
          confidence: 0.7,
          files: ['tests/performance/'],
          metadata: { type: 'monitoring-enhancement' }
        });
      }
      
    } catch (error) {
      console.warn('Performance analysis failed:', error.message);
    }
    
    return items;
  }

  /**
   * Security posture analysis
   */
  discoverFromSecurity() {
    const items = [];
    
    try {
      // Check for hardcoded secrets patterns
      const secretPatterns = execSync('find src/ -type f \\( -name "*.js" -o -name "*.ts" \\) | xargs grep -l "password\\|secret\\|key\\|token" | head -5 || true', { encoding: 'utf8' });
      
      secretPatterns.split('\n').filter(file => file.trim()).forEach(file => {
        items.push({
          id: `sec-${Buffer.from(file).toString('base64').substring(0, 8)}`,
          title: `Review potential secrets in ${path.basename(file)}`,
          type: 'security-review',
          source: 'secret-scan',
          description: `File contains potential hardcoded secrets: ${file}`,
          effort: 2,
          impact: 'high',
          confidence: 0.5,
          files: [file],
          metadata: { file, type: 'secret-detection' }
        });
      });

      // Check Docker configuration security
      if (fs.existsSync('Dockerfile')) {
        items.push({
          id: 'sec-docker',
          title: 'Enhance Docker security configuration',
          type: 'security-enhancement',
          source: 'docker-analysis',
          description: 'Review and harden Docker configuration for production',
          effort: 4,
          impact: 'medium',
          confidence: 0.8,
          files: ['Dockerfile', 'docker-compose.yml'],
          metadata: { type: 'container-security' }
        });
      }
      
    } catch (error) {
      console.warn('Security analysis failed:', error.message);
    }
    
    return items;
  }

  /**
   * Test coverage analysis
   */
  discoverFromTestCoverage() {
    const items = [];
    
    try {
      // Find files without tests
      const srcFiles = execSync('find src/ -name "*.ts" -o -name "*.js" | grep -v ".test." | grep -v ".spec."', { encoding: 'utf8' }).split('\n').filter(f => f.trim());
      const testFiles = execSync('find tests/ -name "*.test.*" -o -name "*.spec.*" 2>/dev/null || echo ""', { encoding: 'utf8' }).split('\n').filter(f => f.trim());
      
      const untestedFiles = srcFiles.filter(srcFile => {
        const baseName = path.basename(srcFile, path.extname(srcFile));
        return !testFiles.some(testFile => testFile.includes(baseName));
      });
      
      untestedFiles.slice(0, 5).forEach(file => {
        items.push({
          id: `test-${Buffer.from(file).toString('base64').substring(0, 8)}`,
          title: `Add tests for ${path.basename(file)}`,
          type: 'testing',
          source: 'coverage-analysis',
          description: `File lacks test coverage: ${file}`,
          effort: 3,
          impact: 'medium',
          confidence: 0.8,
          files: [file],
          metadata: { file, type: 'missing-tests' }
        });
      });
      
    } catch (error) {
      console.warn('Test coverage analysis failed:', error.message);
    }
    
    return items;
  }

  /**
   * Documentation gap analysis
   */
  discoverFromDocumentation() {
    const items = [];
    
    try {
      // Check for API endpoints without documentation
      const apiRoutes = execSync('find src/ -name "*.ts" -o -name "*.js" | xargs grep -l "router\\." | head -5', { encoding: 'utf8' }).split('\n').filter(f => f.trim());
      
      apiRoutes.forEach(file => {
        items.push({
          id: `doc-${Buffer.from(file).toString('base64').substring(0, 8)}`,
          title: `Document API routes in ${path.basename(file)}`,
          type: 'documentation',
          source: 'api-analysis',
          description: `API routes need OpenAPI/Swagger documentation: ${file}`,
          effort: 2,
          impact: 'low',
          confidence: 0.7,
          files: [file],
          metadata: { file, type: 'api-documentation' }
        });
      });
      
      // Check for missing README sections
      if (fs.existsSync('README.md')) {
        const readme = fs.readFileSync('README.md', 'utf8');
        const missingSections = ['Contributing', 'API Reference', 'Deployment'].filter(section => 
          !readme.toLowerCase().includes(section.toLowerCase())
        );
        
        missingSections.forEach(section => {
          items.push({
            id: `readme-${section.toLowerCase()}`,
            title: `Add ${section} section to README`,
            type: 'documentation',
            source: 'readme-analysis',
            description: `README missing ${section} section`,
            effort: 1,
            impact: 'low',
            confidence: 0.9,
            files: ['README.md'],
            metadata: { section, type: 'readme-enhancement' }
          });
        });
      }
      
    } catch (error) {
      console.warn('Documentation analysis failed:', error.message);
    }
    
    return items;
  }

  /**
   * Score work items using WSJF + ICE + Technical Debt model
   */
  scoreWorkItems() {
    const weights = this.config.scoring.weights.maturing;
    
    this.workItems.forEach(item => {
      // WSJF Components
      const userBusinessValue = this.calculateUserBusinessValue(item);
      const timeCriticality = this.calculateTimeCriticality(item);
      const riskReduction = this.calculateRiskReduction(item);
      const opportunityEnablement = this.calculateOpportunityEnablement(item);
      
      const costOfDelay = userBusinessValue + timeCriticality + riskReduction + opportunityEnablement;
      const jobSize = item.effort || 3;
      const wsjfScore = costOfDelay / jobSize;
      
      // ICE Components
      const impact = this.calculateImpact(item);
      const confidence = item.confidence || 0.5;
      const ease = this.calculateEase(item);
      const iceScore = impact * confidence * ease;
      
      // Technical Debt Components
      const debtImpact = this.calculateDebtImpact(item);
      const debtInterest = this.calculateDebtInterest(item);
      const hotspotMultiplier = this.getHotspotMultiplier(item);
      const technicalDebtScore = (debtImpact + debtInterest) * hotspotMultiplier;
      
      // Composite Score
      let compositeScore = (
        weights.wsjf * this.normalizeScore(wsjfScore, 50) +
        weights.ice * this.normalizeScore(iceScore, 10) +
        weights.technicalDebt * this.normalizeScore(technicalDebtScore, 100) +
        weights.security * (item.type.includes('security') ? 20 : 0)
      );
      
      // Apply boosts and penalties
      if (item.type === 'security-fix' || item.type === 'security-update') {
        compositeScore *= this.config.scoring.thresholds.securityBoost;
      }
      if (item.type === 'performance') {
        compositeScore *= (this.config.scoring.thresholds.performanceBoost || 1.5);
      }
      if (item.type === 'documentation') {
        compositeScore *= this.config.scoring.thresholds.documentationPenalty;
      }
      
      item.scores = {
        wsjf: wsjfScore,
        ice: iceScore,
        technicalDebt: technicalDebtScore,
        composite: Math.round(compositeScore * 100) / 100
      };
    });
    
    // Sort by composite score descending
    this.workItems.sort((a, b) => b.scores.composite - a.scores.composite);
    
    return this.workItems;
  }

  /**
   * Calculate user business value (1-10 scale)
   */
  calculateUserBusinessValue(item) {
    const typeValues = {
      'security-fix': 9,
      'security-update': 8,
      'bug-fix': 7,
      'performance': 6,
      'technical-debt': 5,
      'feature': 7,
      'code-quality': 4,
      'testing': 5,
      'documentation': 3,
      'dependency-update': 3
    };
    
    return typeValues[item.type] || 5;
  }

  /**
   * Calculate time criticality (1-10 scale)
   */
  calculateTimeCriticality(item) {
    if (item.type.includes('security')) return 10;
    if (item.type === 'bug-fix') return 8;
    if (item.type === 'performance') return 6;
    if (item.type === 'technical-debt') return 4;
    return 3;
  }

  /**
   * Calculate risk reduction (1-10 scale)
   */
  calculateRiskReduction(item) {
    if (item.type.includes('security')) return 9;
    if (item.type === 'technical-debt') return 7;
    if (item.type === 'testing') return 6;
    if (item.type === 'bug-fix') return 8;
    return 2;
  }

  /**
   * Calculate opportunity enablement (1-10 scale)
   */
  calculateOpportunityEnablement(item) {
    if (item.type === 'performance') return 7;
    if (item.type === 'technical-debt') return 6;
    if (item.type === 'code-quality') return 5;
    if (item.type === 'testing') return 4;
    return 2;
  }

  /**
   * Calculate ICE impact (1-10 scale)
   */
  calculateImpact(item) {
    const impactMap = { 'high': 8, 'medium': 5, 'low': 2 };
    return impactMap[item.impact] || 5;
  }

  /**
   * Calculate ICE ease (1-10 scale)
   */
  calculateEase(item) {
    const effort = item.effort || 3;
    return Math.max(1, 11 - effort); // Invert effort to ease
  }

  /**
   * Calculate technical debt impact
   */
  calculateDebtImpact(item) {
    const baseImpact = item.effort * 2;
    if (item.type === 'technical-debt') return baseImpact * 2;
    if (item.type === 'code-quality') return baseImpact * 1.5;
    return baseImpact;
  }

  /**
   * Calculate technical debt interest (future cost)
   */
  calculateDebtInterest(item) {
    if (item.type === 'technical-debt') return item.effort * 3;
    if (item.type === 'security-fix') return item.effort * 4;
    return item.effort;
  }

  /**
   * Get hotspot multiplier based on file churn
   */
  getHotspotMultiplier(item) {
    if (item.source === 'git-churn') return 3;
    if (item.metadata?.churnCount > 10) return 2.5;
    if (item.metadata?.churnCount > 5) return 2;
    return 1;
  }

  /**
   * Normalize scores to 0-100 scale
   */
  normalizeScore(score, maxValue) {
    return Math.min(100, (score / maxValue) * 100);
  }

  /**
   * Select next best value item for execution
   */
  selectNextBestValue() {
    const candidateItems = this.workItems.filter(item => 
      item.scores.composite >= this.config.scoring.thresholds.minScore
    );
    
    if (candidateItems.length === 0) {
      return this.generateHousekeepingTask();
    }
    
    // Apply additional filters
    for (const item of candidateItems) {
      // Skip if risk exceeds threshold
      if (this.assessRisk(item) > this.config.scoring.thresholds.maxRisk) {
        continue;
      }
      
      // Skip if dependencies not met
      if (!this.areDependenciesMet(item)) {
        continue;
      }
      
      return item;
    }
    
    return this.generateHousekeepingTask();
  }

  /**
   * Generate housekeeping task when no high-value items exist
   */
  generateHousekeepingTask() {
    const housekeepingTasks = [
      {
        id: 'housekeeping-deps',
        title: 'Update development dependencies',
        type: 'maintenance',
        description: 'Update devDependencies to latest versions',
        effort: 2,
        impact: 'low',
        confidence: 0.9
      },
      {
        id: 'housekeeping-docs',
        title: 'Update documentation timestamps',
        type: 'maintenance',
        description: 'Refresh documentation with current information',
        effort: 1,
        impact: 'low',
        confidence: 0.95
      },
      {
        id: 'housekeeping-cleanup',
        title: 'Clean up temporary files and logs',
        type: 'maintenance',
        description: 'Remove temporary files and old log entries',
        effort: 1,
        impact: 'low',
        confidence: 1.0
      }
    ];
    
    return housekeepingTasks[Math.floor(Math.random() * housekeepingTasks.length)];
  }

  /**
   * Assess risk for work item
   */
  assessRisk(item) {
    let risk = 0.1; // Base risk
    
    if (item.type.includes('security')) risk += 0.2;
    if (item.effort > 8) risk += 0.3;
    if (item.confidence < 0.5) risk += 0.4;
    if (item.files?.some(f => f.includes('src/middleware') || f.includes('src/routes'))) risk += 0.2;
    
    return Math.min(1.0, risk);
  }

  /**
   * Check if dependencies are met
   */
  areDependenciesMet(item) {
    // Simple dependency check - in real implementation, this would be more sophisticated
    if (item.type === 'testing' && !fs.existsSync('jest.config.js')) return false;
    if (item.type.includes('security') && !this.hasSecurityTools()) return false;
    return true;
  }

  /**
   * Check if security tools are available
   */
  hasSecurityTools() {
    try {
      execSync('which snyk', { stdio: 'ignore' });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Helper methods for dependency analysis
   */
  isSecurityUpdate(depInfo) {
    // Simple heuristic - major version updates or packages with known security issues
    const current = depInfo.current.split('.').map(Number);
    const latest = depInfo.latest.split('.').map(Number);
    
    return latest[0] > current[0] || // Major version update
           ['express', 'axios', 'lodash'].includes(depInfo.name); // Known security-sensitive packages
  }

  estimateSecurityFixEffort(severity) {
    const effortMap = { 'critical': 8, 'high': 6, 'moderate': 3, 'low': 1 };
    return effortMap[severity] || 3;
  }

  estimateEffortFromComment(comment) {
    if (comment.toLowerCase().includes('refactor')) return 6;
    if (comment.toLowerCase().includes('fix')) return 3;
    if (comment.toLowerCase().includes('add')) return 4;
    return 2;
  }

  assessImpactFromMarker(marker) {
    const impactMap = {
      'FIXME': 'high',
      'BUG': 'high', 
      'HACK': 'medium',
      'TODO': 'medium',
      'DEPRECATED': 'medium',
      'XXX': 'low'
    };
    return impactMap[marker.toUpperCase()] || 'medium';
  }

  /**
   * Generate comprehensive backlog with metrics
   */
  generateBacklog() {
    const topItems = this.workItems.slice(0, 20);
    const averageScore = this.workItems.reduce((sum, item) => sum + item.scores.composite, 0) / this.workItems.length;
    
    const backlog = {
      metadata: {
        lastUpdated: new Date().toISOString(),
        totalItems: this.workItems.length,
        averageScore: averageScore.toFixed(2),
        maturityLevel: this.config.maturity?.level || 65,
        nextExecution: new Date(Date.now() + 30 * 60 * 1000).toISOString() // 30 minutes
      },
      nextBestValue: topItems[0] || null,
      topItems: topItems.map((item, index) => ({
        rank: index + 1,
        id: item.id,
        title: item.title,
        type: item.type,
        source: item.source,
        scores: item.scores,
        effort: item.effort,
        impact: item.impact,
        confidence: item.confidence,
        files: item.files?.slice(0, 3) // Limit files shown
      })),
      categoryBreakdown: this.getCategoryBreakdown(),
      valueMetrics: this.calculateValueMetrics()
    };
    
    return backlog;
  }

  getCategoryBreakdown() {
    const breakdown = {};
    this.workItems.forEach(item => {
      breakdown[item.type] = (breakdown[item.type] || 0) + 1;
    });
    return breakdown;
  }

  calculateValueMetrics() {
    const totalPotentialValue = this.workItems.reduce((sum, item) => sum + item.scores.composite, 0);
    const highValueItems = this.workItems.filter(item => item.scores.composite > 50).length;
    const securityItems = this.workItems.filter(item => item.type.includes('security')).length;
    
    return {
      totalPotentialValue: Math.round(totalPotentialValue),
      highValueItems,
      securityItems,
      averageEffort: (this.workItems.reduce((sum, item) => sum + (item.effort || 3), 0) / this.workItems.length).toFixed(1),
      estimatedWeeksToComplete: Math.ceil(this.workItems.reduce((sum, item) => sum + (item.effort || 3), 0) / 40) // 40 hours per week
    };
  }

  /**
   * Save results to filesystem
   */
  async saveResults() {
    const resultsDir = path.join(__dirname);
    
    // Save detailed backlog
    const backlog = this.generateBacklog();
    fs.writeFileSync(
      path.join(resultsDir, 'value-backlog.json'),
      JSON.stringify(backlog, null, 2)
    );
    
    // Save work items
    fs.writeFileSync(
      path.join(resultsDir, 'work-items.json'),
      JSON.stringify(this.workItems, null, 2)
    );
    
    console.log(`💾 Results saved to ${resultsDir}/`);
    return backlog;
  }

  /**
   * Main execution method
   */
  async run() {
    console.log('🚀 Terragon Autonomous Value Discovery Engine');
    console.log('=' .repeat(50));
    
    await this.discoverWorkItems();
    this.scoreWorkItems();
    const backlog = await this.saveResults();
    
    console.log('\n📊 Discovery Summary:');
    console.log(`Total items discovered: ${this.workItems.length}`);
    console.log(`Average composite score: ${backlog.metadata.averageScore}`);
    console.log(`High-value items (>50): ${backlog.valueMetrics.highValueItems}`);
    console.log(`Security items: ${backlog.valueMetrics.securityItems}`);
    
    if (backlog.nextBestValue) {
      console.log('\n🎯 Next Best Value Item:');
      console.log(`[${backlog.nextBestValue.id}] ${backlog.nextBestValue.title}`);
      console.log(`Composite Score: ${backlog.nextBestValue.scores.composite}`);
      console.log(`Estimated Effort: ${backlog.nextBestValue.effort} hours`);
      console.log(`Type: ${backlog.nextBestValue.type}`);
    }
    
    console.log('\n✨ Value discovery complete!');
    return backlog;
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const engine = new ValueDiscoveryEngine();
  engine.run().catch(console.error);
}

export default ValueDiscoveryEngine;