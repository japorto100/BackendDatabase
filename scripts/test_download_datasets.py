import os
import requests
import tarfile
import zipfile
import argparse
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# Define dataset sources and their download URLs
DATASETS = {
    'rvl-cdip': {
        'url': 'https://www.cs.cmu.edu/~aharley/rvl-cdip/rvl_cdip_sample.zip',
        'description': 'Document classification dataset with 16 classes (sample)',
        'format': 'zip',
        'max_size_mb': 500  # Add size limitation
    },
    'funsd': {
        'url': 'https://guillaumejaume.github.io/FUNSD/dataset.zip',
        'description': 'Form understanding dataset',
        'format': 'zip',
        'max_size_mb': 100
    },
    'docbank': {
        'url': 'https://github.com/doc-analysis/DocBank/raw/master/DocBank_samples.zip',
        'description': 'DocBank sample with token-level annotations',
        'format': 'zip',
        'max_size_mb': 200
    },
    'tobacco800': {
        'url': 'https://www.kaggle.com/datasets/patrickaudriaz/tobacco800-documents',
        'description': 'Tobacco800 document image database',
        'format': 'zip',
        'max_size_mb': 300,
        'kaggle': True  # Flag for Kaggle datasets
    },
    'scitsr': {
        'url': 'https://github.com/Academic-Hammer/SciTSR/archive/refs/heads/master.zip',
        'description': 'Scientific Table Structure Recognition dataset (sample)',
        'format': 'zip',
        'max_size_mb': 100
    },
    'pubtables': {
        'url': 'https://github.com/microsoft/table-transformer/raw/main/data/PubTables-1M-Sample.zip',
        'description': 'PubTables-1M sample for table recognition',
        'format': 'zip',
        'max_size_mb': 200
    }
}

def download_file(url, destination, max_size_mb=None):
    """Download a file with progress bar and size limit"""
    # Check if we need Kaggle credentials
    if url.startswith('https://www.kaggle.com'):
        try:
            import kaggle
            # Extract dataset info from URL
            dataset = url.split('datasets/')[1]
            kaggle.api.dataset_download_files(dataset, path=os.path.dirname(destination), unzip=True)
            return destination
        except ImportError:
            logger.error("Kaggle API not found. Install with: pip install kaggle")
            return None
        except Exception as e:
            logger.error(f"Error downloading from Kaggle: {str(e)}")
            return None
    
    # Regular HTTP download
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Check size limit
    if max_size_mb and total_size > max_size_mb * 1024 * 1024:
        logger.warning(f"File too large: {total_size/(1024*1024):.2f}MB exceeds limit of {max_size_mb}MB")
        return None
    
    block_size = 1024
    
    with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))
    
    return destination

def extract_archive(archive_path, extract_dir, format='zip'):
    """Extract downloaded archive"""
    if format == 'zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif format == 'tar':
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(extract_dir)
    
    return extract_dir

def main():
    parser = argparse.ArgumentParser(description='Download and prepare test document datasets')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS.keys()) + ['all'], 
                        default=['all'], help='Datasets to download')
    parser.add_argument('--output-dir', default='test_documents', 
                        help='Directory to save datasets')
    parser.add_argument('--sample-size', type=int, default=10,
                        help='Number of documents to sample from each dataset')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which datasets to download
    datasets_to_download = list(DATASETS.keys()) if 'all' in args.datasets else args.datasets
    
    for dataset_name in datasets_to_download:
        dataset = DATASETS[dataset_name]
        print(f"Downloading {dataset_name}: {dataset['description']}")
        
        # Create dataset directory
        dataset_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Download dataset with size limit
        archive_path = os.path.join(dataset_dir, f"{dataset_name}.{dataset['format']}")
        result = download_file(
            dataset['url'], 
            archive_path,
            max_size_mb=dataset.get('max_size_mb', 1000)  # Default 1GB limit
        )
        
        if not result:
            print(f"  ✗ Failed to download {dataset_name} - skipping")
            continue
        
        # Extract dataset
        extract_dir = os.path.join(dataset_dir, 'extracted')
        extract_archive(archive_path, extract_dir, dataset['format'])
        
        # Sample documents if needed
        sample_dir = os.path.join(dataset_dir, 'sample')
        os.makedirs(sample_dir, exist_ok=True)
        
        # Find document files
        import glob
        # Document extensions for RVL-CDIP dataset (16 document types)
        doc_extensions = ['.pdf', '.jpg', '.png', '.tif', '.tiff', '.jpeg', '.txt', '.docx', 
                         '.xlsx', '.pptx', '.csv', '.json', '.xml', '.html', '.md', '.rtf']
        documents = []
        for ext in doc_extensions:
            documents.extend(glob.glob(f"{extract_dir}/**/*{ext}", recursive=True))
        
        # Sample documents
        import random
        if len(documents) > args.sample_size:
            documents = random.sample(documents, args.sample_size)
        
        # Copy samples to sample directory
        import shutil
        for doc in documents:
            shutil.copy2(doc, os.path.join(sample_dir, os.path.basename(doc)))
        
        print(f"  ✓ Prepared {len(documents)} sample documents in {sample_dir}")

if __name__ == "__main__":
    main()
