"""URL-based data loader for downloading and extracting financial documents."""

import os
import io
import zipfile
import requests
from typing import List, Optional
from urllib.parse import urlparse
from src.config import get_config

# Optional streamlit import for UI feedback
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None


class URLDataLoader:
    """Handles downloading and extracting data from URLs including Google Drive."""

    @staticmethod
    def is_google_drive_url(url: str) -> bool:
        """Check if the URL is a Google Drive URL."""
        parsed = urlparse(url)
        return parsed.netloc in ['drive.google.com', 'docs.google.com']

    @staticmethod
    def convert_google_drive_url(url: str) -> str:
        """Convert Google Drive URL to direct download link."""
        if 'drive.google.com/file/d/' in url:
            # Extract file ID from URL like: https://drive.google.com/file/d/{FILE_ID}/view
            file_id = url.split('/file/d/')[1].split('/')[0]
            return f'https://drive.google.com/uc?export=download&id={file_id}'
        elif 'drive.google.com/uc?export=download&id=' in url:
            return url  # Already in correct format
        else:
            raise ValueError(f"Unsupported Google Drive URL format: {url}")

    @staticmethod
    def download_file(url: str, timeout: int = 300) -> bytes:
        """Download file from URL with progress tracking."""
        try:
            # Convert Google Drive URL if needed
            if URLDataLoader.is_google_drive_url(url):
                url = URLDataLoader.convert_google_drive_url(url)

            # Download file with progress tracking
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            content = bytearray()

            progress_bar = st.progress(0) if 'st' in globals() else None

            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    content.extend(chunk)
                    downloaded += len(chunk)

                    if progress_bar and total_size > 0:
                        progress = min(downloaded / total_size, 1.0)
                        progress_bar.progress(progress)

            if progress_bar:
                progress_bar.progress(1.0)

            return bytes(content)

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download file from {url}: {str(e)}")
        except Exception as e:
            raise Exception(f"Error downloading file: {str(e)}")

    @staticmethod
    def extract_zip_files(zip_content: bytes, password: Optional[str] = None, extract_to: str = "/tmp") -> List[str]:
        """Extract ZIP file and return list of extracted file paths."""
        try:
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_ref:
                if password:
                    zip_ref.setpassword(password.encode('utf-8'))

                # Create extraction directory if it doesn't exist
                os.makedirs(extract_to, exist_ok=True)

                # Extract all files
                zip_ref.extractall(extract_to)

                # Get list of extracted files (filter for PDFs)
                extracted_files = []
                for root, _, files in os.walk(extract_to):
                    for file in files:
                        if file.lower().endswith('.pdf'):
                            extracted_files.append(os.path.join(root, file))

                return extracted_files

        except zipfile.BadZipFile:
            raise Exception("Downloaded file is not a valid ZIP file")
        except Exception as e:
            raise Exception(f"Failed to extract ZIP file: {str(e)}")

    @staticmethod
    def load_data_from_url(url: str, password: Optional[str] = None, extract_to: str = "/tmp") -> List[str]:
        """
        Load data from URL by downloading and extracting ZIP file.

        Args:
            url: URL to download data from
            password: Optional password for encrypted ZIP files
            extract_to: Directory to extract files to

        Returns:
            List of paths to extracted PDF files
        """
        try:
            # Download the file
            st.info("📥 Downloading data...")
            zip_content = URLDataLoader.download_file(url)

            # Extract ZIP file
            st.info("📂 Extracting ZIP file...")
            extracted_files = URLDataLoader.extract_zip_files(zip_content, password, extract_to)

            if not extracted_files:
                raise Exception("No PDF files found in the downloaded ZIP file")

            st.success(f"✅ Successfully extracted {len(extracted_files)} PDF files")
            return extracted_files

        except Exception as e:
            st.error(f"❌ Failed to load data from URL: {str(e)}")
            raise

    @staticmethod
    def get_document_paths() -> List[str]:
        """
        Get document paths from configuration, with support for URL-based loading.

        Returns:
            List of document paths (local files or downloaded from URL)
        """
        config = get_config()
        document_paths = []

        # If data URL is configured, download and extract documents
        if config.data_url:
            try:
                downloaded_paths = URLDataLoader.load_data_from_url(
                    url=config.data_url,
                    password=config.zip_password,
                    extract_to="/tmp/financial_documents"
                )
                document_paths.extend(downloaded_paths)
            except Exception as e:
                st.error(f"Failed to load documents from URL: {str(e)}")
                # Fall back to local documents if available

        # Add locally configured document paths
        document_paths.extend(config.document_paths)

        return document_paths