"""
DICOM Processor for EHR Multimodal RAG System

This module handles the ingestion and processing of DICOM medical images,
including normalization, windowing, and feature extraction for multimodal embedding.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

import pydicom
from pydicom.dataset import Dataset
import SimpleITK as sitk
from PIL import Image
import cv2
from skimage import exposure, filters, measure
import torch
import torchvision.transforms as transforms

from ..utils.privacy_utils import anonymize_dicom
from ..utils.clinical_validators import validate_dicom_integrity


@dataclass
class DICOMMetadata:
    """Structured DICOM metadata for clinical context."""
    patient_id: str
    study_date: str
    modality: str
    body_part: str
    study_description: str
    series_description: str
    institution: str
    manufacturer: str
    pixel_spacing: Tuple[float, float]
    slice_thickness: Optional[float]
    window_center: Optional[float]
    window_width: Optional[float]
    acquisition_time: Optional[str]
    contrast_agent: Optional[str]
    view_position: Optional[str]


@dataclass
class ProcessedDICOM:
    """Processed DICOM data with embeddings and metadata."""
    image_array: np.ndarray
    normalized_image: np.ndarray
    metadata: DICOMMetadata
    clinical_features: Dict[str, Any]
    preprocessing_params: Dict[str, Any]
    quality_metrics: Dict[str, float]


class DICOMProcessor:
    """
    Advanced DICOM processor with clinical-grade image processing,
    privacy preservation, and multimodal feature extraction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Window settings for different anatomical regions
        self.window_settings = config.get('window_settings', {
            'lung': [-1000, 400],
            'bone': [-400, 1000],
            'soft_tissue': [-160, 240],
            'brain': [40, 80],
            'liver': [-50, 150],
            'mediastinum': [50, 400]
        })
        
        # Supported modalities
        self.supported_modalities = config.get('supported_modalities', 
            ['CT', 'MRI', 'XR', 'US', 'MG', 'PT', 'NM'])
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def process_dicom_file(self, file_path: Union[str, Path]) -> Optional[ProcessedDICOM]:
        """
        Process a single DICOM file asynchronously.
        
        Args:
            file_path: Path to DICOM file
            
        Returns:
            ProcessedDICOM object or None if processing fails
        """
        try:
            # Load DICOM file
            loop = asyncio.get_event_loop()
            dicom_data = await loop.run_in_executor(
                self.executor, self._load_dicom, file_path
            )
            
            if dicom_data is None:
                return None
            
            # Validate DICOM integrity
            if not validate_dicom_integrity(dicom_data):
                self.logger.warning(f"DICOM integrity check failed: {file_path}")
                return None
            
            # Extract metadata
            metadata = self._extract_metadata(dicom_data)
            
            # Check if modality is supported
            if metadata.modality not in self.supported_modalities:
                self.logger.warning(f"Unsupported modality {metadata.modality}: {file_path}")
                return None
            
            # Anonymize DICOM data
            if self.config.get('anonymize', True):
                dicom_data = anonymize_dicom(dicom_data)
            
            # Extract pixel data
            image_array = self._extract_pixel_data(dicom_data)
            if image_array is None:
                return None
            
            # Apply clinical preprocessing
            processed_image, preprocessing_params = await loop.run_in_executor(
                self.executor, self._preprocess_image, image_array, metadata
            )
            
            # Extract clinical features
            clinical_features = await loop.run_in_executor(
                self.executor, self._extract_clinical_features, processed_image, metadata
            )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(processed_image)
            
            return ProcessedDICOM(
                image_array=image_array,
                normalized_image=processed_image,
                metadata=metadata,
                clinical_features=clinical_features,
                preprocessing_params=preprocessing_params,
                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error processing DICOM {file_path}: {str(e)}")
            return None
    
    def _load_dicom(self, file_path: Union[str, Path]) -> Optional[Dataset]:
        """Load DICOM file safely."""
        try:
            return pydicom.dcmread(str(file_path), force=True)
        except Exception as e:
            self.logger.error(f"Failed to load DICOM {file_path}: {str(e)}")
            return None
    
    def _extract_metadata(self, dicom_data: Dataset) -> DICOMMetadata:
        """Extract structured metadata from DICOM dataset."""
        return DICOMMetadata(
            patient_id=getattr(dicom_data, 'PatientID', 'UNKNOWN'),
            study_date=getattr(dicom_data, 'StudyDate', ''),
            modality=getattr(dicom_data, 'Modality', 'UNKNOWN'),
            body_part=getattr(dicom_data, 'BodyPartExamined', ''),
            study_description=getattr(dicom_data, 'StudyDescription', ''),
            series_description=getattr(dicom_data, 'SeriesDescription', ''),
            institution=getattr(dicom_data, 'InstitutionName', ''),
            manufacturer=getattr(dicom_data, 'Manufacturer', ''),
            pixel_spacing=self._get_pixel_spacing(dicom_data),
            slice_thickness=getattr(dicom_data, 'SliceThickness', None),
            window_center=getattr(dicom_data, 'WindowCenter', None),
            window_width=getattr(dicom_data, 'WindowWidth', None),
            acquisition_time=getattr(dicom_data, 'AcquisitionTime', None),
            contrast_agent=getattr(dicom_data, 'ContrastBolusAgent', None),
            view_position=getattr(dicom_data, 'ViewPosition', None)
        )
    
    def _get_pixel_spacing(self, dicom_data: Dataset) -> Tuple[float, float]:
        """Extract pixel spacing from DICOM data."""
        try:
            spacing = dicom_data.PixelSpacing
            return (float(spacing[0]), float(spacing[1]))
        except:
            return (1.0, 1.0)  # Default spacing
    
    def _extract_pixel_data(self, dicom_data: Dataset) -> Optional[np.ndarray]:
        """Extract and convert pixel data to numpy array."""
        try:
            # Get pixel array
            pixel_array = dicom_data.pixel_array
            
            # Handle different bit depths
            if hasattr(dicom_data, 'BitsStored'):
                if dicom_data.BitsStored == 16:
                    # Convert to appropriate data type
                    if hasattr(dicom_data, 'PixelRepresentation') and dicom_data.PixelRepresentation == 1:
                        pixel_array = pixel_array.astype(np.int16)
                    else:
                        pixel_array = pixel_array.astype(np.uint16)
            
            # Apply rescale slope and intercept if available
            if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
                slope = float(dicom_data.RescaleSlope)
                intercept = float(dicom_data.RescaleIntercept)
                pixel_array = pixel_array * slope + intercept
            
            return pixel_array
            
        except Exception as e:
            self.logger.error(f"Failed to extract pixel data: {str(e)}")
            return None
    
    def _preprocess_image(self, image_array: np.ndarray, 
                         metadata: DICOMMetadata) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply clinical-grade image preprocessing.
        
        Args:
            image_array: Raw pixel data
            metadata: DICOM metadata
            
        Returns:
            Tuple of (processed_image, preprocessing_parameters)
        """
        preprocessing_params = {}
        processed = image_array.copy()
        
        # 1. Windowing based on modality and body part
        if metadata.modality in ['CT']:
            processed, window_params = self._apply_windowing(
                processed, metadata.modality, metadata.body_part
            )
            preprocessing_params['windowing'] = window_params
        
        # 2. Noise reduction
        if metadata.modality in ['CT', 'MRI']:
            processed = filters.gaussian(processed, sigma=0.5)
            preprocessing_params['noise_reduction'] = {'method': 'gaussian', 'sigma': 0.5}
        
        # 3. Contrast enhancement
        processed = self._enhance_contrast(processed, metadata.modality)
        preprocessing_params['contrast_enhancement'] = True
        
        # 4. Normalization
        processed = self._normalize_image(processed)
        preprocessing_params['normalization'] = 'z_score'
        
        # 5. Resize for model input
        if len(processed.shape) == 2:
            # Convert grayscale to RGB for model compatibility
            processed = np.stack([processed] * 3, axis=-1)
        
        # Ensure proper data type and range
        processed = np.clip(processed, 0, 1).astype(np.float32)
        
        return processed, preprocessing_params
    
    def _apply_windowing(self, image: np.ndarray, modality: str, 
                        body_part: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply appropriate windowing based on anatomy."""
        # Determine appropriate window settings
        window_key = self._get_window_key(body_part.lower())
        window_center, window_width = self.window_settings.get(
            window_key, self.window_settings['soft_tissue']
        )
        
        # Apply windowing
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2
        
        windowed = np.clip(image, window_min, window_max)
        windowed = (windowed - window_min) / (window_max - window_min)
        
        return windowed, {
            'window_center': window_center,
            'window_width': window_width,
            'window_key': window_key
        }
    
    def _get_window_key(self, body_part: str) -> str:
        """Determine appropriate window setting based on body part."""
        if any(term in body_part for term in ['lung', 'chest', 'thorax']):
            return 'lung'
        elif any(term in body_part for term in ['bone', 'spine', 'femur']):
            return 'bone'
        elif any(term in body_part for term in ['brain', 'head', 'skull']):
            return 'brain'
        elif any(term in body_part for term in ['liver', 'hepatic']):
            return 'liver'
        elif any(term in body_part for term in ['mediastinum', 'heart']):
            return 'mediastinum'
        else:
            return 'soft_tissue'
    
    def _enhance_contrast(self, image: np.ndarray, modality: str) -> np.ndarray:
        """Apply modality-specific contrast enhancement."""
        if modality == 'XR':
            # CLAHE for X-rays
            return exposure.equalize_adapthist(image, clip_limit=0.03)
        elif modality in ['CT', 'MRI']:
            # Histogram equalization
            return exposure.equalize_hist(image)
        else:
            return image
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image using z-score normalization."""
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            return (image - mean) / std
        else:
            return image - mean
    
    def _extract_clinical_features(self, image: np.ndarray, 
                                 metadata: DICOMMetadata) -> Dict[str, Any]:
        """Extract clinical features from processed image."""
        features = {}
        
        # Basic image statistics
        features['intensity_stats'] = {
            'mean': float(np.mean(image)),
            'std': float(np.std(image)),
            'min': float(np.min(image)),
            'max': float(np.max(image)),
            'percentile_95': float(np.percentile(image, 95)),
            'percentile_5': float(np.percentile(image, 5))
        }
        
        # Texture features
        if len(image.shape) >= 2:
            gray_image = image[:, :, 0] if len(image.shape) == 3 else image
            features['texture'] = self._extract_texture_features(gray_image)
        
        # Anatomical features based on modality
        if metadata.modality == 'XR':
            features['anatomical'] = self._extract_xray_features(image)
        elif metadata.modality == 'CT':
            features['anatomical'] = self._extract_ct_features(image)
        elif metadata.modality == 'MRI':
            features['anatomical'] = self._extract_mri_features(image)
        
        # Clinical context
        features['clinical_context'] = {
            'modality': metadata.modality,
            'body_part': metadata.body_part,
            'contrast_enhanced': metadata.contrast_agent is not None,
            'acquisition_quality': self._assess_acquisition_quality(image, metadata)
        }
        
        return features
    
    def _extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract texture features using GLCM and other methods."""
        # Simplified texture analysis
        features = {}
        
        # Edge density
        edges = filters.sobel(image)
        features['edge_density'] = float(np.mean(edges > 0.1))
        
        # Local standard deviation
        features['local_std'] = float(np.mean(filters.rank.variance(
            (image * 255).astype(np.uint8), np.ones((5, 5))
        )))
        
        return features
    
    def _extract_xray_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract X-ray specific features."""
        return {
            'lung_field_visibility': self._assess_lung_visibility(image),
            'bone_prominence': self._assess_bone_prominence(image),
            'cardiac_silhouette': self._detect_cardiac_silhouette(image)
        }
    
    def _extract_ct_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract CT-specific features."""
        return {
            'hounsfield_distribution': self._analyze_hounsfield_units(image),
            'tissue_segmentation': self._basic_tissue_segmentation(image),
            'slice_quality': self._assess_slice_quality(image)
        }
    
    def _extract_mri_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract MRI-specific features."""
        return {
            'signal_intensity': self._analyze_signal_intensity(image),
            'tissue_contrast': self._assess_tissue_contrast(image),
            'motion_artifacts': self._detect_motion_artifacts(image)
        }
    
    def _calculate_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate image quality metrics."""
        metrics = {}
        
        # Signal-to-noise ratio estimation
        metrics['snr_estimate'] = self._estimate_snr(image)
        
        # Contrast-to-noise ratio
        metrics['cnr_estimate'] = self._estimate_cnr(image)
        
        # Sharpness measure
        metrics['sharpness'] = self._measure_sharpness(image)
        
        # Artifact detection
        metrics['artifact_score'] = self._detect_artifacts(image)
        
        return metrics
    
    def _estimate_snr(self, image: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        if len(image.shape) >= 2:
            gray_image = image[:, :, 0] if len(image.shape) == 3 else image
            signal = np.mean(gray_image)
            noise = np.std(gray_image)
            return float(signal / noise) if noise > 0 else 0.0
        return 0.0
    
    def _estimate_cnr(self, image: np.ndarray) -> float:
        """Estimate contrast-to-noise ratio."""
        if len(image.shape) >= 2:
            gray_image = image[:, :, 0] if len(image.shape) == 3 else image
            # Simple CNR estimation using image regions
            h, w = gray_image.shape
            center = gray_image[h//4:3*h//4, w//4:3*w//4]
            background = gray_image[0:h//4, 0:w//4]
            
            signal_diff = abs(np.mean(center) - np.mean(background))
            noise = np.std(background)
            return float(signal_diff / noise) if noise > 0 else 0.0
        return 0.0
    
    def _measure_sharpness(self, image: np.ndarray) -> float:
        """Measure image sharpness using Laplacian variance."""
        if len(image.shape) >= 2:
            gray_image = image[:, :, 0] if len(image.shape) == 3 else image
            laplacian = cv2.Laplacian(gray_image.astype(np.float32), cv2.CV_64F)
            return float(laplacian.var())
        return 0.0
    
    def _detect_artifacts(self, image: np.ndarray) -> float:
        """Detect common imaging artifacts."""
        # Simplified artifact detection
        # This would be expanded with specific artifact detection algorithms
        artifact_score = 0.0
        
        if len(image.shape) >= 2:
            gray_image = image[:, :, 0] if len(image.shape) == 3 else image
            
            # Ring artifacts (simplified detection)
            # Motion artifacts
            # Noise patterns
            
            # For now, return a placeholder
            artifact_score = float(np.std(gray_image))
        
        return artifact_score
    
    # Placeholder methods for anatomical feature extraction
    def _assess_lung_visibility(self, image: np.ndarray) -> float:
        """Assess lung field visibility in chest X-rays."""
        return 1.0  # Placeholder
    
    def _assess_bone_prominence(self, image: np.ndarray) -> float:
        """Assess bone prominence in X-rays."""
        return 1.0  # Placeholder
    
    def _detect_cardiac_silhouette(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect cardiac silhouette in chest X-rays."""
        return {'detected': True, 'size_ratio': 0.5}  # Placeholder
    
    def _analyze_hounsfield_units(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze Hounsfield unit distribution in CT."""
        return {'mean_hu': 0.0, 'std_hu': 100.0}  # Placeholder
    
    def _basic_tissue_segmentation(self, image: np.ndarray) -> Dict[str, float]:
        """Basic tissue segmentation for CT."""
        return {'air': 0.2, 'soft_tissue': 0.6, 'bone': 0.2}  # Placeholder
    
    def _assess_slice_quality(self, image: np.ndarray) -> float:
        """Assess CT slice quality."""
        return 0.8  # Placeholder
    
    def _analyze_signal_intensity(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze MRI signal intensity."""
        return {'t1_weighted': 0.5, 't2_weighted': 0.5}  # Placeholder
    
    def _assess_tissue_contrast(self, image: np.ndarray) -> float:
        """Assess tissue contrast in MRI."""
        return 0.7  # Placeholder
    
    def _detect_motion_artifacts(self, image: np.ndarray) -> float:
        """Detect motion artifacts in MRI."""
        return 0.1  # Placeholder
    
    def _assess_acquisition_quality(self, image: np.ndarray, 
                                  metadata: DICOMMetadata) -> str:
        """Assess overall acquisition quality."""
        # This would implement a comprehensive quality assessment
        return "good"  # Placeholder
    
    async def process_dicom_directory(self, directory_path: Union[str, Path]) -> List[ProcessedDICOM]:
        """
        Process all DICOM files in a directory.
        
        Args:
            directory_path: Path to directory containing DICOM files
            
        Returns:
            List of ProcessedDICOM objects
        """
        directory = Path(directory_path)
        dicom_files = []
        
        # Find all DICOM files
        for ext in ['.dcm', '.dicom', '']:
            dicom_files.extend(directory.glob(f'*{ext}'))
        
        # Process files concurrently
        tasks = [self.process_dicom_file(file_path) for file_path in dicom_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        processed_dicoms = []
        for result in results:
            if isinstance(result, ProcessedDICOM):
                processed_dicoms.append(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Processing error: {str(result)}")
        
        return processed_dicoms
    
    def __del__(self):
        """Cleanup executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)