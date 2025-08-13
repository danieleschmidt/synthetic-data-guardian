"""
Text Data Generator - Synthetic text data generation
"""

import asyncio
import time
import random
import string
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from .base import BaseGenerator, GeneratorConfig, GenerationResult
from ..utils.logger import get_logger


@dataclass
class TextGeneratorConfig(GeneratorConfig):
    """Configuration for text generator."""
    backend: str = "template"  # template, markov, llm
    min_length: int = 10
    max_length: int = 100
    templates: List[str] = None
    vocabulary: List[str] = None
    patterns: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.templates is None:
            self.templates = [
                "This is a sample text with {placeholder}.",
                "Generated content for {purpose} purposes.",
                "Random text data: {value}"
            ]
        if self.vocabulary is None:
            self.vocabulary = ["sample", "test", "data", "synthetic", "generated"]
        if self.patterns is None:
            self.patterns = ["email", "phone", "name", "address"]


class TextGenerator(BaseGenerator):
    """Simple text data generator with template-based generation."""
    
    def __init__(self, config: Optional[TextGeneratorConfig] = None, logger=None):
        if config is None:
            config = TextGeneratorConfig(name="text_generator", type="text")
        super().__init__(config, logger)
    
    async def initialize(self) -> None:
        if self.initialized:
            return
        self.logger.info("Initializing TextGenerator...")
        self.initialized = True
        self.logger.info("TextGenerator initialized")
    
    async def generate(self, num_records: int, seed: Optional[int] = None, **kwargs) -> GenerationResult:
        if seed:
            random.seed(seed)
        
        start_time = time.time()
        texts = []
        
        # Get templates and vocabulary with fallbacks
        templates = getattr(self.config, 'templates', [
            "This is a sample text with {placeholder}.",
            "Generated content for {purpose} purposes.",
            "Random text data: {value}"
        ])
        vocabulary = getattr(self.config, 'vocabulary', ["sample", "test", "data", "synthetic", "generated"])
        
        for _ in range(num_records):
            template = random.choice(templates)
            text = template.format(
                placeholder=random.choice(vocabulary),
                purpose=random.choice(["testing", "development", "research"]),
                value=random.randint(1, 1000)
            )
            texts.append(text)
        
        generation_time = time.time() - start_time
        await self._update_statistics(generation_time, num_records)
        
        return GenerationResult(
            data=texts,
            metadata={
                'generation_time': generation_time,
                'num_records': num_records,
                'backend': 'template'
            }
        )
    
    def validate_config(self) -> List[str]:
        return []
    
    async def cleanup(self) -> None:
        self.initialized = False