"""
StegaStamp Watermarker - Image watermarking using steganography techniques
"""

import asyncio
import time
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from .base import BaseWatermarker
from ..utils.logger import get_logger


class StegaStampWatermarker(BaseWatermarker):
    """Simple steganographic watermarker for images and data."""
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        super().__init__(config, logger)
        self.strength = self.config.get('strength', 0.1)
    
    async def initialize(self) -> None:
        if self.initialized:
            return
        self.logger.info("Initializing StegaStampWatermarker...")
        self.initialized = True  
        self.logger.info("StegaStampWatermarker initialized")
    
    async def embed(
        self,
        data: Any,
        message: str,
        key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Embed steganographic watermark."""
        start_time = time.time()
        
        try:
            # Simple LSB-style embedding for demonstration
            if isinstance(data, np.ndarray) and data.dtype == np.uint8:
                # Image data
                watermarked_data = self._embed_lsb(data, message, key)
            else:
                # Fallback to statistical embedding
                watermarked_data = self._embed_statistical(data, message, key)
            
            embedding_time = time.time() - start_time
            
            return {
                'data': watermarked_data,
                'watermark_embedded': True,
                'method': 'stegastamp',
                'strength': self.strength,
                'embedding_time': embedding_time,
                'message_hash': hashlib.sha256(message.encode()).hexdigest()[:16]
            }
            
        except Exception as e:
            self.logger.error(f"StegaStamp embedding failed: {str(e)}")
            return {
                'data': data,
                'watermark_embedded': False,
                'error': str(e)
            }
    
    async def extract(
        self,
        data: Any,
        key: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Extract steganographic watermark."""
        try:
            if isinstance(data, np.ndarray) and data.dtype == np.uint8:
                return self._extract_lsb(data, key)
            else:
                # For non-image data, return verification result
                verification = await self.verify(data, key=key)
                if verification.get('is_watermarked', False):
                    return True, "stegastamp_watermark_detected"
                return False, None
                
        except Exception as e:
            self.logger.error(f"StegaStamp extraction failed: {str(e)}")
            return False, None
    
    async def verify(
        self,
        data: Any,
        expected_message: Optional[str] = None,
        key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Verify steganographic watermark."""
        start_time = time.time()
        
        try:
            is_watermarked, extracted_message = await self.extract(data, key)
            
            if expected_message and extracted_message:
                message_match = expected_message in extracted_message
            else:
                message_match = is_watermarked
            
            verification_time = time.time() - start_time
            
            return {
                'is_watermarked': is_watermarked,
                'message_match': message_match,
                'extracted_message': extracted_message,
                'confidence': 0.9 if is_watermarked else 0.0,
                'verification_time': verification_time
            }
            
        except Exception as e:
            self.logger.error(f"StegaStamp verification failed: {str(e)}")
            return {
                'is_watermarked': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _embed_lsb(self, image_data: np.ndarray, message: str, key: Optional[str]) -> np.ndarray:
        """Embed message using LSB steganography."""
        try:
            watermarked = image_data.copy()
            
            # Convert message to binary
            message_with_delimiter = message + "|||END|||"
            binary_message = ''.join(format(ord(char), '08b') for char in message_with_delimiter)
            
            # Flatten image for easier processing
            flat_image = watermarked.flatten()
            
            if len(binary_message) > len(flat_image):
                self.logger.warning("Message too long for image capacity")
                return image_data
            
            # Embed message bits in LSBs
            for i, bit in enumerate(binary_message):
                if i < len(flat_image):
                    # Clear LSB and set to message bit
                    flat_image[i] = (flat_image[i] & 0xFE) | int(bit)
            
            return flat_image.reshape(image_data.shape)
            
        except Exception as e:
            self.logger.warning(f"LSB embedding failed: {str(e)}")
            return image_data
    
    def _extract_lsb(self, image_data: np.ndarray, key: Optional[str]) -> Tuple[bool, Optional[str]]:
        """Extract message using LSB steganography."""
        try:
            flat_image = image_data.flatten()
            
            # Extract LSBs
            binary_message = ""
            for pixel in flat_image:
                binary_message += str(pixel & 1)
            
            # Convert binary to text
            message = ""
            for i in range(0, len(binary_message), 8):
                byte = binary_message[i:i+8]
                if len(byte) == 8:
                    char = chr(int(byte, 2))
                    message += char
                    
                    # Check for delimiter
                    if message.endswith("|||END|||"):
                        extracted_message = message[:-9]  # Remove delimiter
                        return True, extracted_message
            
            return False, None
            
        except Exception as e:
            self.logger.warning(f"LSB extraction failed: {str(e)}")
            return False, None
    
    def _embed_statistical(self, data: Any, message: str, key: Optional[str]) -> Any:
        """Fallback statistical embedding for non-image data."""
        try:
            # Use a simple approach similar to statistical watermarker
            signature = self._generate_signature(message, key)
            
            if hasattr(data, 'copy') and hasattr(data, 'select_dtypes'):
                # DataFrame
                watermarked_data = data.copy()
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                
                for column in numeric_columns:
                    col_std = data[column].std()
                    if col_std > 0:
                        # Apply small modification
                        modification = signature * col_std * self.strength
                        watermarked_data[column] = data[column] + modification
                
                return watermarked_data
            else:
                return data
                
        except Exception as e:
            self.logger.warning(f"Statistical embedding fallback failed: {str(e)}")
            return data
    
    def _generate_signature(self, message: str, key: Optional[str]) -> float:
        """Generate signature from message and key."""
        content = f"{message}_{key or 'default_key'}"
        hash_digest = hashlib.md5(content.encode()).hexdigest()
        hash_int = int(hash_digest[:8], 16)
        return (hash_int / (2**32 - 1)) * 2 - 1
    
    async def cleanup(self) -> None:
        self.initialized = False
        self.logger.info("StegaStampWatermarker cleanup completed")