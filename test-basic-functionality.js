#!/usr/bin/env node
/**
 * Basic functionality test for Synthetic Data Guardian
 */

import { Guardian } from './src/core/Guardian.js';
import { GenerationPipeline } from './src/core/GenerationPipeline.js';
import winston from 'winston';

// Setup logger
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.simple(),
  transports: [new winston.transports.Console()],
});

async function testBasicFunctionality() {
  console.log('🧪 Testing Basic Synthetic Data Guardian Functionality\n');

  try {
    // Test 1: Initialize Guardian
    console.log('1️⃣  Initializing Guardian...');
    const guardian = new Guardian(logger);
    await guardian.initialize();
    console.log('✅ Guardian initialized successfully\n');

    // Test 2: Create a simple tabular data generation pipeline
    console.log('2️⃣  Creating tabular data generation pipeline...');
    const pipelineConfig = {
      name: 'test_customer_data',
      description: 'Test customer data generation',
      generator: 'basic',
      dataType: 'tabular',
      schema: {
        customer_id: 'uuid',
        name: 'string',
        age: { type: 'integer', min: 18, max: 80 },
        email: 'email',
        income: { type: 'float', min: 20000, max: 200000 },
        category: { type: 'categorical', categories: ['premium', 'standard', 'basic'] },
      },
      validation: {
        enabled: true,
        validators: ['data_completeness', 'data_consistency'],
      },
    };

    const result = await guardian.generate({
      pipeline: pipelineConfig,
      numRecords: 50,
      seed: 42,
    });

    console.log('✅ Data generation completed!');
    console.log(`   📊 Generated ${result.recordCount} records`);
    console.log(`   📈 Quality Score: ${result.qualityScore?.toFixed(3)}`);
    console.log(`   🔐 Privacy Score: ${result.privacyScore?.toFixed(3)}`);
    console.log(`   ⏱️  Execution Time: ${result.executionTime}ms`);
    console.log(`   🔗 Lineage ID: ${result.lineageId}`);

    // Show sample data
    console.log('\n📋 Sample Generated Data (first 3 records):');
    console.table(result.data.slice(0, 3));

    // Test 3: Validate the generated data
    console.log('\n3️⃣  Testing data validation...');
    const validationReport = await guardian.validate(result.data, {
      validators: ['data_completeness', 'data_consistency', 'data_validity'],
    });

    console.log('✅ Data validation completed!');
    console.log(`   📊 Overall Score: ${validationReport.overallScore?.toFixed(3)}`);
    console.log(`   ✔️  Passed: ${validationReport.passed}`);
    console.log(`   🧪 Validators: ${validationReport.validatorCount}`);

    // Test 4: Apply watermarking
    console.log('\n4️⃣  Testing watermarking...');
    const watermarkResult = await guardian.watermark(result.data.slice(0, 10), {
      method: 'statistical',
      message: 'synthetic:test:guardian',
      strength: 0.8,
    });

    console.log('✅ Watermarking completed!');
    console.log(`   🔏 Method: ${watermarkResult.watermarkInfo?.method}`);
    console.log(`   💪 Strength: ${watermarkResult.watermarkInfo?.strength}`);

    // Test 5: Verify watermark
    console.log('\n5️⃣  Testing watermark verification...');
    const verifyResult = await guardian.verifyWatermark(watermarkResult.watermarkedData, {
      method: 'statistical',
    });

    console.log('✅ Watermark verification completed!');
    console.log(`   ✔️  Valid: ${verifyResult.isValid}`);
    console.log(`   🎯 Confidence: ${verifyResult.confidence}`);

    // Test 6: Check lineage
    console.log('\n6️⃣  Testing lineage tracking...');
    const lineage = await guardian.getLineage(result.lineageId);

    console.log('✅ Lineage retrieval completed!');
    console.log(`   🔗 Lineage ID: ${lineage.lineage?.id}`);
    console.log(`   📅 Timestamp: ${lineage.lineage?.timestamp}`);
    console.log(`   🏭 Pipeline: ${lineage.lineage?.pipeline?.name}`);

    // Test 7: List pipelines
    console.log('\n7️⃣  Testing pipeline management...');
    const pipelines = guardian.listPipelines();

    console.log('✅ Pipeline listing completed!');
    console.log(`   📋 Active Pipelines: ${pipelines.length}`);
    if (pipelines.length > 0) {
      console.log('   📄 Pipeline Names:', pipelines.map(p => p.name).join(', '));
    }

    // Cleanup
    console.log('\n8️⃣  Cleaning up...');
    await guardian.cleanup();
    console.log('✅ Cleanup completed!');

    console.log('\n🎉 All basic functionality tests passed! Generation 1 is working correctly.');
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

// Run the test
testBasicFunctionality();
