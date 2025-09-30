"""
Google Cloud Storage utilities for the stock prediction system.
"""

import os
from pathlib import Path
from typing import Optional, List, Union

try:
    import pandas as pd
    from google.cloud import storage
    from google.cloud.exceptions import NotFound
except ImportError:
    # Handle case where packages are not installed yet
    pd = None
    storage = None
    NotFound = None

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class GCSManager:
    """Manager for Google Cloud Storage operations."""
    
    def __init__(self, project_id: str, bucket_name: str, credentials_path: Optional[str] = None):
        """
        Initialize GCS manager.
        
        Args:
            project_id: GCP project ID
            bucket_name: GCS bucket name
            credentials_path: Path to service account credentials JSON file
        """
        self.project_id = project_id
        self.bucket_name = bucket_name
        
        # Set up credentials
        if credentials_path and Path(credentials_path).exists():
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        try:
            self.client = storage.Client(project=project_id)
            self.bucket = self.client.bucket(bucket_name)
            logger.info(f"Connected to GCS bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"Failed to connect to GCS: {e}")
            raise
    
    def create_bucket_if_not_exists(self, location: str = "US-CENTRAL1") -> bool:
        """
        Create bucket if it doesn't exist.
        
        Args:
            location: Bucket location
            
        Returns:
            True if bucket was created or already exists, False otherwise
        """
        try:
            self.bucket.reload()
            logger.info(f"Bucket {self.bucket_name} already exists")
            return True
        except NotFound:
            try:
                bucket = self.client.create_bucket(self.bucket_name, location=location)
                logger.info(f"Created bucket {bucket.name} in {location}")
                self.bucket = bucket
                return True
            except Exception as e:
                logger.error(f"Failed to create bucket {self.bucket_name}: {e}")
                return False
    
    def upload_file(self, local_path: Union[str, Path], blob_name: str) -> bool:
        """
        Upload a file to GCS.
        
        Args:
            local_path: Path to local file
            blob_name: Name for the blob in GCS
            
        Returns:
            True if successful, False otherwise
        """
        try:
            blob = self.bucket.blob(blob_name)
            local_path = Path(local_path)
            
            blob.upload_from_filename(str(local_path))
            logger.info(f"Uploaded {local_path} to {blob_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to {blob_name}: {e}")
            return False
    
    def download_file(self, blob_name: str, local_path: Union[str, Path]) -> bool:
        """
        Download a file from GCS.
        
        Args:
            blob_name: Name of the blob in GCS
            local_path: Local path to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            blob = self.bucket.blob(blob_name)
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            blob.download_to_filename(str(local_path))
            logger.info(f"Downloaded {blob_name} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {blob_name} to {local_path}: {e}")
            return False
    
    def upload_dataframe(self, df: pd.DataFrame, blob_name: str, format: str = "parquet") -> bool:
        """
        Upload a pandas DataFrame to GCS.
        
        Args:
            df: Pandas DataFrame to upload
            blob_name: Name for the blob in GCS
            format: File format ('parquet', 'csv', 'json')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            blob = self.bucket.blob(blob_name)
            
            if format.lower() == "parquet":
                # Upload parquet directly
                parquet_buffer = df.to_parquet()
                blob.upload_from_string(parquet_buffer, content_type="application/octet-stream")
            elif format.lower() == "csv":
                csv_buffer = df.to_csv(index=False)
                blob.upload_from_string(csv_buffer, content_type="text/csv")
            elif format.lower() == "json":
                json_buffer = df.to_json(orient="records", date_format="iso")
                blob.upload_from_string(json_buffer, content_type="application/json")
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Uploaded DataFrame to {blob_name} ({format} format)")
            return True
        except Exception as e:
            logger.error(f"Failed to upload DataFrame to {blob_name}: {e}")
            return False
    
    def download_dataframe(self, blob_name: str, format: str = "parquet") -> Optional[pd.DataFrame]:
        """
        Download a pandas DataFrame from GCS.
        
        Args:
            blob_name: Name of the blob in GCS
            format: File format ('parquet', 'csv', 'json')
            
        Returns:
            Pandas DataFrame if successful, None otherwise
        """
        try:
            blob = self.bucket.blob(blob_name)
            data = blob.download_as_text() if format != "parquet" else blob.download_as_bytes()
            
            if format.lower() == "parquet":
                df = pd.read_parquet(data)
            elif format.lower() == "csv":
                df = pd.read_csv(data)
            elif format.lower() == "json":
                df = pd.read_json(data, orient="records")
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Downloaded DataFrame from {blob_name} ({format} format)")
            return df
        except Exception as e:
            logger.error(f"Failed to download DataFrame from {blob_name}: {e}")
            return None
    
    def list_blobs(self, prefix: str = "") -> List[str]:
        """
        List all blobs in the bucket with the given prefix.
        
        Args:
            prefix: Prefix to filter blobs
            
        Returns:
            List of blob names
        """
        try:
            blobs = self.client.list_blobs(self.bucket, prefix=prefix)
            blob_names = [blob.name for blob in blobs]
            logger.info(f"Found {len(blob_names)} blobs with prefix '{prefix}'")
            return blob_names
        except Exception as e:
            logger.error(f"Failed to list blobs with prefix '{prefix}': {e}")
            return []
    
    def delete_blob(self, blob_name: str) -> bool:
        """
        Delete a blob from GCS.
        
        Args:
            blob_name: Name of the blob to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            blob = self.bucket.blob(blob_name)
            blob.delete()
            logger.info(f"Deleted blob {blob_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete blob {blob_name}: {e}")
            return False
    
    def blob_exists(self, blob_name: str) -> bool:
        """
        Check if a blob exists in GCS.
        
        Args:
            blob_name: Name of the blob to check
            
        Returns:
            True if blob exists, False otherwise
        """
        try:
            blob = self.bucket.blob(blob_name)
            return blob.exists()
        except Exception as e:
            logger.error(f"Failed to check if blob {blob_name} exists: {e}")
            return False