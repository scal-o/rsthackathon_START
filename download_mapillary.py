#!/usr/bin/env python3
"""
Mapillary Image Downloader

Downloads all Mapillary images within a specified radius around a location.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests


class MapillaryDownloader:
    """Download Mapillary images for a specific location."""

    API_URL = "https://graph.mapillary.com/images"

    def __init__(self, api_key: str, output_dir: str = "downloads"):
        """
        Initialize the downloader.

        Args:
            api_key: Mapillary API key
            output_dir: Directory to save downloaded images
        """
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def calculate_bbox(self, lat: float, lon: float, radius_km: float) -> tuple:
        """
        Calculate bounding box from center point and radius.

        Args:
            lat: Latitude of center point
            lon: Longitude of center point
            radius_km: Radius in kilometers

        Returns:
            Tuple of (min_lon, min_lat, max_lon, max_lat)
        """
        # Approximate degrees per km (varies by latitude)
        import math

        lat_deg_per_km = 1 / 110.574
        lon_deg_per_km = 1 / (111.320 * abs(math.cos(math.radians(lat))))

        lat_offset = radius_km * lat_deg_per_km
        lon_offset = radius_km * lon_deg_per_km

        return (
            lon - lon_offset,  # min_lon
            lat - lat_offset,  # min_lat
            lon + lon_offset,  # max_lon
            lat + lat_offset,  # max_lat
        )

    def fetch_images_metadata(
        self, lat: float, lon: float, radius_km: float, limit: int = 2000
    ) -> List[Dict]:
        """
        Fetch metadata for all images in the area.

        Args:
            lat: Latitude of center point
            lon: Longitude of center point
            radius_km: Search radius in kilometers
            limit: Maximum number of images to fetch

        Returns:
            List of image metadata dictionaries
        """
        bbox = self.calculate_bbox(lat, lon, radius_km)
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

        print(f"Searching in bounding box: {bbox_str}")
        print(f"Center: ({lat}, {lon}), Radius: {radius_km} km")

        all_images = []
        page_token = None

        while len(all_images) < limit:
            params = {
                "access_token": self.api_key,
                "fields": "id,thumb_1024_url,thumb_2048_url,captured_at,compass_angle,geometry",
                "bbox": bbox_str,
                "limit": min(1000, limit - len(all_images)),
            }

            if page_token:
                params["after"] = page_token

            try:
                response = requests.get(self.API_URL, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()
                images = data.get("data", [])

                if not images:
                    break

                all_images.extend(images)
                print(f"Fetched {len(all_images)} images so far...")

                # Check if there's another page
                paging = data.get("paging", {})
                if "cursors" in paging and "after" in paging["cursors"]:
                    page_token = paging["cursors"]["after"]
                else:
                    break

                # Rate limiting
                time.sleep(0.5)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching images: {e}")
                break

        print(f"Total images found: {len(all_images)}")
        return all_images

    def download_image(self, image_data: Dict, use_high_res: bool = False) -> Optional[str]:
        """
        Download a single image.

        Args:
            image_data: Image metadata dictionary
            use_high_res: If True, download 2048px version instead of 1024px

        Returns:
            Path to downloaded file or None if failed
        """
        image_id = image_data["id"]

        # Choose resolution
        url_key = "thumb_2048_url" if use_high_res else "thumb_1024_url"
        image_url = image_data.get(url_key)

        if not image_url:
            print(f"No URL found for image {image_id}")
            return None

        # Create filename with timestamp
        captured_at = image_data.get("captured_at", "unknown")
        # Ensure captured_at is a string
        if not isinstance(captured_at, str):
            captured_at = str(captured_at)
        filename = f"{image_id}_{captured_at.replace(':', '-')}.jpg"
        filepath = self.output_dir / filename

        # Skip if already downloaded
        if filepath.exists():
            return str(filepath)

        try:
            response = requests.get(image_url, timeout=30, stream=True)
            response.raise_for_status()

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return str(filepath)

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {image_id}: {e}")
            return None

    def download_all(
        self,
        lat: float,
        lon: float,
        radius_km: float,
        max_images: int = 2000,
        high_res: bool = False,
    ) -> None:
        """
        Download all images in the specified area.

        Args:
            lat: Latitude of center point
            lon: Longitude of center point
            radius_km: Search radius in kilometers
            max_images: Maximum number of images to download
            high_res: Download 2048px images instead of 1024px
        """
        print(f"\n{'=' * 60}")
        print(f"Mapillary Image Downloader")
        print(f"{'=' * 60}\n")

        # Fetch metadata
        print("Step 1: Fetching image metadata...")
        images = self.fetch_images_metadata(lat, lon, radius_km, max_images)

        if not images:
            print("No images found in this area.")
            return

        # Save metadata
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(images, f, indent=2)
        print(f"Metadata saved to {metadata_file}")

        # Download images
        print(f"\nStep 2: Downloading {len(images)} images...")
        print(f"Resolution: {'2048px' if high_res else '1024px'}")
        print(f"Output directory: {self.output_dir}\n")

        downloaded = 0
        failed = 0

        for i, image_data in enumerate(images, 1):
            print(f"[{i}/{len(images)}] Downloading {image_data['id']}...", end=" ")

            result = self.download_image(image_data, high_res)

            if result:
                downloaded += 1
                print("✓")
            else:
                failed += 1
                print("✗")

            # Rate limiting
            time.sleep(0.3)

        print(f"\n{'=' * 60}")
        print(f"Download complete!")
        print(f"Downloaded: {downloaded}")
        print(f"Failed: {failed}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download Mapillary images for a specific location",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download images around Paris (Eiffel Tower) within 0.5 km
  python download_mapillary.py 48.8584 2.2945 0.5
  
  # Download with custom API key and output directory
  python download_mapillary.py 48.8584 2.2945 1.0 --api-key YOUR_KEY --output paris_images
  
  # Download high resolution images (2048px)
  python download_mapillary.py 48.8584 2.2945 0.5 --high-res
  
  # Limit to 500 images
  python download_mapillary.py 48.8584 2.2945 2.0 --max-images 500
        """,
    )

    parser.add_argument("latitude", type=float, help="Latitude of center point")
    parser.add_argument("longitude", type=float, help="Longitude of center point")
    parser.add_argument("radius", type=float, help="Search radius in kilometers")

    parser.add_argument(
        "--api-key", type=str, help="Mapillary API key (or set MAPILLARY_API_KEY env variable)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="mapillary_downloads",
        help="Output directory (default: mapillary_downloads)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=2000,
        help="Maximum number of images to download (default: 2000)",
    )
    parser.add_argument(
        "--high-res", action="store_true", help="Download 2048px images instead of 1024px"
    )

    args = parser.parse_args()

    # Get API key from multiple sources (priority order)
    api_key = args.api_key or os.environ.get("MAPILLARY_API_KEY")

    # Try to read from Streamlit secrets if available
    if not api_key:
        try:
            from streamlit import secrets

            api_key = secrets.get("MAPILLARY_API_KEY")
        except (ImportError, FileNotFoundError, KeyError):
            pass

    if not api_key:
        print("Error: Mapillary API key required!")
        print("Provide via:")
        print("  1. --api-key argument")
        print("  2. MAPILLARY_API_KEY environment variable")
        print("  3. .streamlit/secrets.toml file")
        print("\nGet a free API key at: https://www.mapillary.com/developer")
        sys.exit(1)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output}_{timestamp}"

    # Initialize downloader
    downloader = MapillaryDownloader(api_key, output_dir)

    # Download images
    try:
        downloader.download_all(
            args.latitude, args.longitude, args.radius, args.max_images, args.high_res
        )
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
