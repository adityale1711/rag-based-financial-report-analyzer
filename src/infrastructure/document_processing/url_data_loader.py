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
    def download_file(url: str, timeout: int = 60) -> bytes:
        """Download file from URL with progress tracking and robust error handling."""
        try:
            # Convert Google Drive URL if needed
            if URLDataLoader.is_google_drive_url(url):
                url = URLDataLoader.convert_google_drive_url(url)

            # Configure headers to avoid potential issues
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            # Download file with progress tracking and better error handling
            response = requests.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True)
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

        except requests.exceptions.Timeout:
            raise Exception(f"Download timeout after {timeout} seconds. The file might be too large or the server is slow to respond.")
        except requests.exceptions.ConnectionError:
            raise Exception(f"Connection error while downloading from {url}. Please check the URL and your internet connection.")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP error {e.response.status_code} while downloading from {url}. The file might not be publicly accessible.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download file from {url}: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error downloading file: {str(e)}")

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
            # Validate URL before attempting download
            if not url or not url.strip():
                raise Exception("URL is empty or invalid")

            # Show download status
            if STREAMLIT_AVAILABLE and st:
                st.info("📥 Downloading data from URL...")

            # Download the file with shorter timeout for cloud deployment
            zip_content = URLDataLoader.download_file(url, timeout=30)

            if not zip_content:
                raise Exception("Downloaded file is empty")

            # Show extraction status
            if STREAMLIT_AVAILABLE and st:
                st.info("📂 Extracting ZIP file...")

            # Extract ZIP file
            extracted_files = URLDataLoader.extract_zip_files(zip_content, password, extract_to)

            if not extracted_files:
                raise Exception("No PDF files found in the downloaded ZIP file")

            # Show success status
            if STREAMLIT_AVAILABLE and st:
                st.success(f"✅ Successfully extracted {len(extracted_files)} PDF files")

            return extracted_files

        except Exception as e:
            # Show error status 
            if STREAMLIT_AVAILABLE and st:
                st.error(f"❌ Failed to load data from URL: {str(e)}")

            # log the error but not crash the app
            print(f"URL Data Loading Error: {str(e)}")
            
            # Return empty list instead of raising to allow fallback to local files
            return []

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

                # Log successful URL loading
                print(f"Successfully loaded {len(downloaded_paths)} documents from URL")
            except Exception as e:
                # Log error but don't crash - fall back to local documents
                print(f"Warning: Failed to load documents from URL: {str(e)}")
                if STREAMLIT_AVAILABLE and st:
                    st.warning(f"⚠️ Could not load documents from URL, falling back to local files")

        # Add locally configured document paths
        document_paths.extend(config.document_paths)

        # Filter to only existing files to avoid errors
        existing_paths = []
        for path in document_paths:
            if os.path.exists(path):
                existing_paths.append(path)
            else:
                print(f"Warning: Document not found: {path}")

        # If no documents found and we were trying to load from URL, provide a helpful message
        if not existing_paths and config.data_url:
            if STREAMLIT_AVAILABLE and st:
                st.error("❌ No financial documents found. The URL download may have failed or the documents may not be accessible.")
                st.info("💡 Please check your Google Drive sharing settings or ensure the ZIP file contains PDF documents.")

        return existing_paths