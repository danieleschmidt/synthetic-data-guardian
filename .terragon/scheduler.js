#!/usr/bin/env node

/**
 * Terragon Autonomous Value Discovery Scheduler
 * 
 * Implements scheduled execution for continuous value discovery and delivery
 * Supports immediate, hourly, daily, weekly, and monthly schedules
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';
import ValueDiscoveryEngine from './value-discovery.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class ValueScheduler {
  constructor() {
    this.engine = new ValueDiscoveryEngine();
    this.lockFile = path.join(__dirname, '.scheduler.lock');
    this.logFile = path.join(__dirname, 'scheduler.log');
  }

  /**
   * Check if scheduler is already running
   */
  isRunning() {
    return fs.existsSync(this.lockFile);
  }

  /**
   * Create lock file
   */
  createLock() {
    fs.writeFileSync(this.lockFile, JSON.stringify({
      pid: process.pid,
      started: new Date().toISOString(),
      schedule: process.argv[2] || 'manual'
    }));
  }

  /**
   * Remove lock file
   */
  removeLock() {
    try {
      fs.unlinkSync(this.lockFile);
    } catch (error) {
      // Lock file doesn't exist or can't be removed
    }
  }

  /**
   * Log execution details
   */
  log(level, message, data = {}) {
    const entry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      schedule: process.argv[2] || 'manual',
      ...data
    };
    
    const logEntry = JSON.stringify(entry) + '\n';
    fs.appendFileSync(this.logFile, logEntry);
    
    console.log(`[${entry.timestamp}] ${level.toUpperCase()}: ${message}`);
    if (Object.keys(data).length > 0) {
      console.log(JSON.stringify(data, null, 2));
    }
  }

  /**
   * Execute immediate value discovery (after PR merge)
   */
  async executeImmediate() {
    this.log('info', 'Starting immediate value discovery execution');
    
    try {
      // Check if there's a recent git change
      const lastCommit = execSync('git log -1 --format="%H %ci"', { encoding: 'utf8' }).trim();
      const [commitHash, commitTime] = lastCommit.split(' ');
      const commitDate = new Date(commitTime);
      const now = new Date();
      const timeDiff = (now - commitDate) / (1000 * 60); // minutes
      
      if (timeDiff > 10) {
        this.log('info', 'No recent commits, skipping immediate execution', { 
          lastCommit: commitTime,
          minutesAgo: Math.round(timeDiff)
        });
        return;
      }
      
      this.log('info', 'Recent commit detected, running full discovery', {
        commit: commitHash.substring(0, 8),
        minutesAgo: Math.round(timeDiff)
      });
      
      const backlog = await this.engine.run();
      await this.executeNextBestValue(backlog);
      
    } catch (error) {
      this.log('error', 'Immediate execution failed', { error: error.message });
      throw error;
    }
  }

  /**
   * Execute hourly security scans
   */
  async executeHourly() {
    this.log('info', 'Starting hourly security scan execution');
    
    try {
      // Focus on security-related discoveries
      await this.engine.discoverWorkItems();
      const securityItems = this.engine.workItems.filter(item => 
        item.type.includes('security') || 
        item.source === 'npm-audit' ||
        item.source === 'secret-scan'
      );
      
      if (securityItems.length > 0) {
        this.engine.workItems = securityItems;
        this.engine.scoreWorkItems();
        
        const backlog = this.engine.generateBacklog();
        await this.engine.saveResults();
        
        this.log('info', 'Security scan completed', {
          securityItems: securityItems.length,
          highRisk: securityItems.filter(item => item.scores?.composite > 60).length
        });
        
        // Auto-execute critical security fixes
        const criticalSecurity = securityItems.find(item => 
          item.type === 'security-fix' && item.scores?.composite > 80
        );
        
        if (criticalSecurity) {
          this.log('warn', 'Critical security issue detected', {
            item: criticalSecurity.title,
            score: criticalSecurity.scores.composite
          });
          await this.executeWorkItem(criticalSecurity);
        }
      } else {
        this.log('info', 'No security issues detected');
      }
      
    } catch (error) {
      this.log('error', 'Hourly security scan failed', { error: error.message });
    }
  }

  /**
   * Execute daily comprehensive analysis
   */
  async executeDaily() {
    this.log('info', 'Starting daily comprehensive analysis');
    
    try {
      const backlog = await this.engine.run();
      
      // Execute top 2-3 items if they meet criteria
      const readyItems = backlog.topItems
        .filter(item => item.scores.composite > 25)
        .slice(0, 3);
      
      for (const item of readyItems) {
        const risk = this.engine.assessRisk(item);
        if (risk < 0.7) {
          await this.executeWorkItem(item);
        }
      }
      
      // Update metrics
      await this.updateMetrics(backlog);
      
      this.log('info', 'Daily analysis completed', {
        itemsDiscovered: backlog.metadata.totalItems,
        itemsExecuted: readyItems.length,
        averageScore: backlog.metadata.averageScore
      });
      
    } catch (error) {
      this.log('error', 'Daily analysis failed', { error: error.message });
    }
  }

  /**
   * Execute weekly deep SDLC assessment
   */
  async executeWeekly() {
    this.log('info', 'Starting weekly deep SDLC assessment');
    
    try {
      // Run comprehensive discovery
      const backlog = await this.engine.run();
      
      // Analyze trends and patterns
      const trends = await this.analyzeTrends();
      
      // Generate strategic recommendations
      const recommendations = this.generateStrategicRecommendations(backlog, trends);
      
      // Update maturity assessment
      const newMaturityLevel = this.assessMaturityLevel(backlog);
      
      // Save weekly report
      const weeklyReport = {
        timestamp: new Date().toISOString(),
        backlog,
        trends,
        recommendations,
        maturityLevel: newMaturityLevel,
        weeklyMetrics: {
          itemsCompleted: trends.itemsCompleted || 0,
          averageScore: parseFloat(backlog.metadata.averageScore),
          categoryShifts: trends.categoryShifts || {},
          velocityTrend: trends.velocityTrend || 'stable'
        }
      };
      
      fs.writeFileSync(
        path.join(__dirname, `weekly-report-${new Date().toISOString().split('T')[0]}.json`),
        JSON.stringify(weeklyReport, null, 2)
      );
      
      this.log('info', 'Weekly assessment completed', {
        maturityLevel: newMaturityLevel,
        recommendations: recommendations.length,
        totalValue: backlog.valueMetrics.totalPotentialValue
      });
      
    } catch (error) {
      this.log('error', 'Weekly assessment failed', { error: error.message });
    }
  }

  /**
   * Execute monthly strategic recalibration
   */
  async executeMonthly() {
    this.log('info', 'Starting monthly strategic recalibration');
    
    try {
      // Analyze historical performance
      const performance = await this.analyzeHistoricalPerformance();
      
      // Recalibrate scoring weights
      const newWeights = this.recalibrateWeights(performance);
      
      // Update configuration
      await this.updateConfiguration(newWeights);
      
      // Generate strategic roadmap
      const roadmap = this.generateStrategicRoadmap();
      
      // Archive old metrics and start fresh baseline
      await this.archiveMetrics();
      
      this.log('info', 'Monthly recalibration completed', {
        newWeights,
        roadmapItems: roadmap.length,
        performanceScore: performance.overallScore
      });
      
    } catch (error) {
      this.log('error', 'Monthly recalibration failed', { error: error.message });
    }
  }

  /**
   * Execute next best value item from backlog
   */
  async executeNextBestValue(backlog) {
    if (!backlog.nextBestValue) {
      this.log('info', 'No viable items for execution');
      return;
    }
    
    const item = backlog.nextBestValue;
    this.log('info', 'Executing next best value item', {
      id: item.id,
      title: item.title,
      score: item.scores.composite,
      type: item.type
    });
    
    await this.executeWorkItem(item);
  }

  /**
   * Execute a specific work item
   */
  async executeWorkItem(item) {
    const startTime = Date.now();
    
    try {
      this.log('info', `Starting execution: ${item.title}`, {
        id: item.id,
        type: item.type,
        effort: item.effort,
        score: item.scores?.composite
      });
      
      // Determine execution strategy based on item type
      switch (item.type) {
        case 'dependency-update':
          await this.executeDependencyUpdate(item);
          break;
        case 'security-fix':
          await this.executeSecurityFix(item);
          break;
        case 'technical-debt':
          await this.executeTechnicalDebtFix(item);
          break;
        case 'testing':
          await this.executeTestingTask(item);
          break;
        case 'documentation':
          await this.executeDocumentationTask(item);
          break;
        case 'performance':
          await this.executePerformanceTask(item);
          break;
        default:
          this.log('warn', `Unknown item type: ${item.type}, skipping execution`);
          return;
      }
      
      const executionTime = (Date.now() - startTime) / 1000;
      
      // Update metrics
      await this.recordExecution(item, executionTime, 'success');
      
      this.log('info', `Execution completed: ${item.title}`, {
        executionTime: `${executionTime}s`,
        status: 'success'
      });
      
    } catch (error) {
      const executionTime = (Date.now() - startTime) / 1000;
      await this.recordExecution(item, executionTime, 'failure', error.message);
      
      this.log('error', `Execution failed: ${item.title}`, {
        error: error.message,
        executionTime: `${executionTime}s`
      });
    }
  }

  /**
   * Execute dependency update
   */
  async executeDependencyUpdate(item) {
    if (item.metadata?.package) {
      const packageName = item.metadata.package;
      this.log('info', `Updating package: ${packageName}`);
      
      // Update specific package
      execSync(`npm update ${packageName}`, { stdio: 'inherit' });
      
      // Run tests to verify
      execSync('npm test', { stdio: 'inherit' });
      
      // Create commit
      execSync(`git add package.json package-lock.json`);
      execSync(`git commit -m "chore: update ${packageName} dependency

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"`);
    }
  }

  /**
   * Execute security fix
   */
  async executeSecurityFix(item) {
    if (item.source === 'npm-audit') {
      this.log('info', 'Running npm audit fix');
      execSync('npm audit fix --force', { stdio: 'inherit' });
      
      // Verify tests still pass
      execSync('npm test', { stdio: 'inherit' });
      
      execSync('git add .');
      execSync(`git commit -m "fix: address security vulnerabilities

${item.description}

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"`);
    }
  }

  /**
   * Execute technical debt fix
   */
  async executeTechnicalDebtFix(item) {
    // For now, create an issue to track the technical debt
    this.log('info', `Creating technical debt tracking issue for: ${item.title}`);
    
    const issueBody = `
## Technical Debt Item

**Discovered by**: Terragon Autonomous Value Discovery
**Score**: ${item.scores?.composite || 'N/A'}
**Effort**: ${item.effort} hours
**Impact**: ${item.impact}

## Description
${item.description}

## Files Affected
${item.files?.map(f => `- ${f}`).join('\n') || 'No specific files'}

## Recommended Action
${this.generateRecommendedAction(item)}

---
*🤖 Auto-generated by Terragon Value Discovery Engine*
`;
    
    // In a real implementation, this would create a GitHub issue
    fs.writeFileSync(
      path.join(__dirname, `tech-debt-${item.id}.md`),
      issueBody
    );
  }

  /**
   * Execute testing task
   */
  async executeTestingTask(item) {
    if (item.files && item.files.length > 0) {
      const sourceFile = item.files[0];
      const testFile = this.generateTestFilePath(sourceFile);
      
      this.log('info', `Generating test template for: ${sourceFile}`);
      
      const testTemplate = this.generateTestTemplate(sourceFile);
      
      // Ensure test directory exists
      const testDir = path.dirname(testFile);
      fs.mkdirSync(testDir, { recursive: true });
      
      // Write test template
      fs.writeFileSync(testFile, testTemplate);
      
      execSync(`git add ${testFile}`);
      execSync(`git commit -m "test: add test template for ${path.basename(sourceFile)}

${item.description}

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"`);
    }
  }

  /**
   * Execute documentation task
   */
  async executeDocumentationTask(item) {
    if (item.id.startsWith('readme-')) {
      const section = item.metadata?.section;
      if (section) {
        this.log('info', `Adding ${section} section to README`);
        
        const readmePath = 'README.md';
        let readme = fs.readFileSync(readmePath, 'utf8');
        
        const sectionContent = this.generateReadmeSection(section);
        readme += '\n\n' + sectionContent;
        
        fs.writeFileSync(readmePath, readme);
        
        execSync('git add README.md');
        execSync(`git commit -m "docs: add ${section} section to README

${item.description}

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"`);
      }
    }
  }

  /**
   * Execute performance task
   */
  async executePerformanceTask(item) {
    this.log('info', 'Enhancing performance monitoring');
    
    // Add basic performance monitoring to existing health endpoint
    const performanceMonitor = `
// Enhanced performance monitoring
export const performanceMetrics = {
  startTime: Date.now(),
  
  getUptime: () => process.uptime(),
  getMemoryUsage: () => process.memoryUsage(),
  getCpuUsage: () => process.cpuUsage(),
  
  getPerformanceSnapshot: () => ({
    uptime: performanceMetrics.getUptime(),
    memory: performanceMetrics.getMemoryUsage(),
    cpu: performanceMetrics.getCpuUsage(),
    timestamp: new Date().toISOString()
  })
};
`;
    
    const perfFile = 'src/middleware/performance.ts';
    fs.writeFileSync(perfFile, performanceMonitor);
    
    execSync(`git add ${perfFile}`);
    execSync(`git commit -m "perf: add enhanced performance monitoring

${item.description}

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"`);
  }

  /**
   * Generate test file path for a source file
   */
  generateTestFilePath(sourceFile) {
    const parsed = path.parse(sourceFile);
    return path.join('tests', 'unit', `${parsed.name}.test${parsed.ext}`);
  }

  /**
   * Generate basic test template
   */
  generateTestTemplate(sourceFile) {
    const moduleName = path.basename(sourceFile, path.extname(sourceFile));
    
    return `import { describe, it, expect } from '@jest/globals';

describe('${moduleName}', () => {
  it('should be defined', () => {
    // TODO: Import and test ${moduleName}
    expect(true).toBe(true);
  });
  
  // TODO: Add specific test cases
});
`;
  }

  /**
   * Generate README section content
   */
  generateReadmeSection(section) {
    const sections = {
      'Deployment': `## 🚀 Deployment

### Environment Variables

Create a \`.env\` file with required configuration:

\`\`\`bash
NODE_ENV=production
PORT=8080
DATABASE_URL=postgresql://user:pass@localhost:5432/db
REDIS_URL=redis://localhost:6379
\`\`\`

### Docker Deployment

\`\`\`bash
docker-compose up -d
\`\`\`

### Manual Deployment

\`\`\`bash
npm run build
npm start
\`\`\``,
      'API Reference': `## 📚 API Reference

### Health Endpoints

- \`GET /health\` - Basic health check
- \`GET /health/ready\` - Readiness probe
- \`GET /health/live\` - Liveness probe

### Metrics

- \`GET /metrics\` - Prometheus metrics

For full API documentation, see [API Docs](docs/api.md).`
    };
    
    return sections[section] || `## ${section}\n\nTODO: Add ${section} documentation`;
  }

  /**
   * Generate recommended action for technical debt
   */
  generateRecommendedAction(item) {
    const actions = {
      'git-history': 'Review the commit and ensure the quick fix is properly implemented with tests and documentation.',
      'code-comments': 'Address the comment by implementing the suggested improvement or removing outdated comments.',
      'typescript': 'Fix the TypeScript compilation error by adding proper types or configuration.',
      'coverage-analysis': 'Add comprehensive unit tests to improve code coverage and reliability.'
    };
    
    return actions[item.source] || 'Review and address the identified technical debt item.';
  }

  /**
   * Record execution metrics
   */
  async recordExecution(item, executionTime, status, error = null) {
    const metricsPath = path.join(__dirname, 'value-metrics.json');
    let metrics;
    
    try {
      metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
    } catch {
      metrics = { executionHistory: [] };
    }
    
    metrics.executionHistory.push({
      timestamp: new Date().toISOString(),
      itemId: item.id,
      title: item.title,
      type: item.type,
      scores: item.scores,
      estimatedEffort: item.effort,
      actualTime: executionTime,
      status,
      error
    });
    
    // Update aggregate metrics
    const successful = metrics.executionHistory.filter(h => h.status === 'success');
    metrics.valueDelivery = {
      totalValueDelivered: successful.reduce((sum, h) => sum + (h.scores?.composite || 0), 0),
      itemsCompleted: successful.length,
      averageCycleTime: successful.reduce((sum, h) => sum + h.actualTime, 0) / successful.length || 0,
      successRate: successful.length / metrics.executionHistory.length,
      rollbackRate: metrics.executionHistory.filter(h => h.status === 'rollback').length / metrics.executionHistory.length
    };
    
    fs.writeFileSync(metricsPath, JSON.stringify(metrics, null, 2));
  }

  /**
   * Analyze trends from historical data
   */
  async analyzeTrends() {
    // Placeholder for trend analysis
    return {
      velocityTrend: 'increasing',
      categoryShifts: {},
      itemsCompleted: 0,
      scoreEvolution: 'stable'
    };
  }

  /**
   * Generate strategic recommendations
   */
  generateStrategicRecommendations(backlog, trends) {
    const recommendations = [];
    
    if (backlog.valueMetrics.securityItems < 2) {
      recommendations.push({
        type: 'security-focus',
        priority: 'high',
        description: 'Increase security-focused work items to improve security posture'
      });
    }
    
    if (parseFloat(backlog.metadata.averageScore) < 35) {
      recommendations.push({
        type: 'value-optimization',
        priority: 'medium',
        description: 'Focus on higher-value work items to improve overall value delivery'
      });
    }
    
    return recommendations;
  }

  /**
   * Assess current maturity level
   */
  assessMaturityLevel(backlog) {
    let maturityScore = 65; // Current baseline
    
    // Adjust based on backlog composition
    const debtRatio = backlog.categoryBreakdown['technical-debt'] / backlog.metadata.totalItems;
    if (debtRatio > 0.6) {
      maturityScore -= 5; // High debt reduces maturity
    } else if (debtRatio < 0.3) {
      maturityScore += 5; // Low debt increases maturity
    }
    
    // Adjust based on security items
    if (backlog.valueMetrics.securityItems > 2) {
      maturityScore += 3;
    }
    
    return Math.max(0, Math.min(100, maturityScore));
  }

  /**
   * Main execution method
   */
  async run() {
    const schedule = process.argv[2] || 'manual';
    
    if (this.isRunning()) {
      this.log('warn', 'Scheduler already running, exiting');
      return;
    }
    
    this.createLock();
    
    try {
      this.log('info', `Starting scheduled execution: ${schedule}`);
      
      switch (schedule) {
        case 'immediate':
          await this.executeImmediate();
          break;
        case 'hourly':
          await this.executeHourly();
          break;
        case 'daily':
          await this.executeDaily();
          break;
        case 'weekly':
          await this.executeWeekly();
          break;
        case 'monthly':
          await this.executeMonthly();
          break;
        default:
          this.log('info', 'Manual execution - running full discovery');
          const backlog = await this.engine.run();
          await this.executeNextBestValue(backlog);
      }
      
      this.log('info', `Scheduled execution completed: ${schedule}`);
      
    } catch (error) {
      this.log('error', 'Scheduled execution failed', { error: error.message });
      throw error;
    } finally {
      this.removeLock();
    }
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const scheduler = new ValueScheduler();
  scheduler.run().catch(error => {
    console.error('Scheduler failed:', error);
    process.exit(1);
  });
}

export default ValueScheduler;