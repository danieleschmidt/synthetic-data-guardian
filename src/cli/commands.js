/**
 * CLI Commands - Command-line interface handlers
 */

import { readFileSync, writeFileSync } from 'fs';
import { Guardian } from '../core/Guardian.js';
import { GenerationPipeline } from '../core/GenerationPipeline.js';
import winston from 'winston';

// Setup CLI logger
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.colorize(),
    winston.format.simple()
  ),
  transports: [new winston.transports.Console()]
});

export async function handleCommand(command, args) {
  try {
    switch (command) {
      case 'generate':
        await handleGenerate(args);
        break;
      case 'validate':
        await handleValidate(args);
        break;
      case 'lineage':
        await handleLineage(args);
        break;
      case 'watermark':
        await handleWatermark(args);
        break;
      default:
        console.error(`Unknown command: ${command}`);
        process.exit(1);
    }
  } catch (error) {
    console.error('Command failed:', error.message);
    process.exit(1);
  }
}

async function handleGenerate(args) {
  const configFile = args[0];
  if (!configFile) {
    console.error('Usage: generate <config-file>');
    process.exit(1);
  }

  const outputDir = args.find(arg => arg.startsWith('--output='))?.split('=')[1] || './output';
  
  // Read configuration
  const configContent = readFileSync(configFile, 'utf8');
  const config = JSON.parse(configContent);

  // Initialize Guardian
  const guardian = new Guardian(logger);
  await guardian.initialize();

  // Generate data
  console.log('Starting data generation...');
  const result = await guardian.generate(config);

  // Save result
  const outputFile = `${outputDir}/synthetic_data_${result.taskId}.json`;
  writeFileSync(outputFile, JSON.stringify(result.data, null, 2));

  console.log(`Generated ${result.recordCount} records`);
  console.log(`Quality Score: ${result.qualityScore}`);
  console.log(`Privacy Score: ${result.privacyScore}`);
  console.log(`Output saved to: ${outputFile}`);

  await guardian.cleanup();
}

async function handleValidate(args) {
  const dataFile = args[0];
  const schemaFile = args[1];

  if (!dataFile || !schemaFile) {
    console.error('Usage: validate <data-file> <schema-file>');
    process.exit(1);
  }

  // Read data and schema
  const data = JSON.parse(readFileSync(dataFile, 'utf8'));
  const schema = JSON.parse(readFileSync(schemaFile, 'utf8'));

  // Initialize Guardian
  const guardian = new Guardian(logger);
  await guardian.initialize();

  // Validate data
  console.log('Validating data...');
  const report = await guardian.validate(data, { schema });

  console.log(`Validation ${report.passed ? 'PASSED' : 'FAILED'}`);
  console.log(`Overall Score: ${report.overallScore}`);
  
  for (const result of report.validationResults) {
    console.log(`${result.validator}: ${result.passed ? 'PASS' : 'FAIL'} (${result.score})`);
  }

  await guardian.cleanup();
}

async function handleLineage(args) {
  const lineageId = args[0];
  if (!lineageId) {
    console.error('Usage: lineage <lineage-id>');
    process.exit(1);
  }

  // Initialize Guardian
  const guardian = new Guardian(logger);
  await guardian.initialize();

  // Get lineage
  try {
    const lineage = await guardian.getLineage(lineageId);
    console.log('Lineage Information:');
    console.log(JSON.stringify(lineage, null, 2));
  } catch (error) {
    console.error('Lineage not found:', error.message);
  }

  await guardian.cleanup();
}

async function handleWatermark(args) {
  const dataFile = args[0];
  if (!dataFile) {
    console.error('Usage: watermark <data-file>');
    process.exit(1);
  }

  const outputFile = args.find(arg => arg.startsWith('--output='))?.split('=')[1] || 'watermarked_data.json';

  // Read data
  const data = JSON.parse(readFileSync(dataFile, 'utf8'));

  // Initialize Guardian
  const guardian = new Guardian(logger);
  await guardian.initialize();

  // Apply watermark
  console.log('Applying watermark...');
  const result = await guardian.watermark(data);

  // Save watermarked data
  writeFileSync(outputFile, JSON.stringify(result.watermarkedData, null, 2));

  console.log('Watermark applied successfully');
  console.log(`Method: ${result.watermarkInfo.method}`);
  console.log(`Output saved to: ${outputFile}`);

  await guardian.cleanup();
}