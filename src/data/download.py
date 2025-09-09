"""
Kaggle Dataset Download Module

This module handles downloading medical imaging datasets from Kaggle.
Supports COVID-19 CT scans, lung cancer, brain tumor, and pneumonia datasets.
"""

import os
import zipfile
import argparse
from pathlib import Path
from typing import Optional

# Optional Kaggle import (requires authentication)
try:
    import kaggle
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except (ImportError, OSError) as e:
    KAGGLE_AVAILABLE = False
    print(f"ℹ️  Kaggle API not available: {e}")
    print("   Install kaggle and set up credentials to use Kaggle datasets")
    print("   Project can still work with online streaming and synthetic data")


class KaggleDatasetDownloader:
    """Downloads and manages Kaggle datasets for medical imaging projects."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the dataset downloader.
        
        Args:
            data_dir: Directory to store downloaded datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kaggle API only if available
        if KAGGLE_AVAILABLE:
            self.api = KaggleApi()
            self.api.authenticate()
        else:
            self.api = None
            print("⚠️ Kaggle API not available - dataset downloads disabled")
            print("   Use online streaming or synthetic data instead")
        
        # Available datasets
        self.datasets = {
            'covid19-ct': 'maedemaftouni/large-covid19-ct-slice-dataset',
            'covid19-scans': 'andrewmvd/covid19-ct-scans',
            'lung-cancer': 'mohamedhanyyy/chest-ctscan-images',
            'lung-cancer-iq': 'hamdallak/the-iq-oth-nccd-lung-cancer-dataset',
            'brain-tumor-mri': 'sartajbhuvaji/brain-tumor-classification-mri',
            'brain-tumor-detection': 'navoneel/brain-mri-images-for-brain-tumor-detection',
            'pneumonia-xray': 'paultimothymooney/chest-xray-pneumonia',
            'covid19-radiography': 'tawsifurrahman/covid19-radiography-database'
        }
    
    def list_available_datasets(self) -> None:
        """Print all available datasets."""
        print("Available datasets:")
        print("-" * 50)
        for key, dataset in self.datasets.items():
            print(f"{key:20} : {dataset}")
    
    def download_dataset(self, dataset_key: str, unzip: bool = True) -> Path:
        """
        Download a dataset from Kaggle.
        
        Args:
            dataset_key: Key from self.datasets dictionary
            unzip: Whether to automatically unzip the dataset
            
        Returns:
            Path to the downloaded dataset directory
        """
        if not KAGGLE_AVAILABLE or self.api is None:
            raise RuntimeError("Kaggle API not available. Use online streaming instead.")
            
        if dataset_key not in self.datasets:
            raise ValueError(f"Dataset '{dataset_key}' not found. Available: {list(self.datasets.keys())}")
        
        dataset_name = self.datasets[dataset_key]
        dataset_dir = self.data_dir / dataset_key
        dataset_dir.mkdir(exist_ok=True)
        
        print(f"Downloading {dataset_name} to {dataset_dir}")
        
        # Download dataset
        self.api.dataset_download_files(
            dataset_name,
            path=str(dataset_dir),
            unzip=unzip
        )
        
        print(f"Successfully downloaded {dataset_key}")
        return dataset_dir
    
    def download_competition_data(self, competition: str) -> Path:
        """
        Download competition data.
        
        Args:
            competition: Kaggle competition name
            
        Returns:
            Path to the downloaded competition directory
        """
        comp_dir = self.data_dir / f"competition_{competition}"
        comp_dir.mkdir(exist_ok=True)
        
        print(f"Downloading competition {competition} to {comp_dir}")
        
        self.api.competition_download_files(
            competition,
            path=str(comp_dir),
            unzip=True
        )
        
        print(f"Successfully downloaded competition {competition}")
        return comp_dir
    
    def get_dataset_info(self, dataset_key: str) -> dict:
        """
        Get information about a dataset.
        
        Args:
            dataset_key: Key from self.datasets dictionary
            
        Returns:
            Dataset information dictionary
        """
        if dataset_key not in self.datasets:
            raise ValueError(f"Dataset '{dataset_key}' not found.")
        
        dataset_name = self.datasets[dataset_key]
        dataset_info = self.api.dataset_view(dataset_name)
        
        return {
            'title': dataset_info.title,
            'size': dataset_info.totalBytes,
            'files': dataset_info.files,
            'description': dataset_info.description,
            'url': dataset_info.url
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Download Kaggle datasets for medical imaging')
    parser.add_argument('--dataset', type=str, help='Dataset key to download')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Data directory')
    parser.add_argument('--info', type=str, help='Get information about a dataset')
    parser.add_argument('--competition', type=str, help='Download competition data')
    
    args = parser.parse_args()
    
    downloader = KaggleDatasetDownloader(args.data_dir)
    
    if args.list:
        downloader.list_available_datasets()
    elif args.info:
        info = downloader.get_dataset_info(args.info)
        print(f"Dataset: {info['title']}")
        print(f"Size: {info['size']} bytes")
        print(f"Files: {len(info['files'])}")
        print(f"Description: {info['description'][:200]}...")
    elif args.dataset:
        downloader.download_dataset(args.dataset)
    elif args.competition:
        downloader.download_competition_data(args.competition)
    else:
        print("Please specify --dataset, --list, --info, or --competition")


if __name__ == "__main__":
    main()
