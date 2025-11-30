import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import folium
import requests
import streamlit as st
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium

# Page config
st.set_page_config(page_title="City Map Dashboard with Inference", page_icon="🗺️", layout="wide")

# Mapillary API configuration
MAPILLARY_API_KEY = st.secrets.get("MAPILLARY_API_KEY", "YOUR_API_KEY_HERE")
MAPILLARY_API_URL = "https://graph.mapillary.com/images"

# Paths
GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "data", "points.geojson")
IMAGES_PRE_PATH = Path(__file__).parent / "static" / "images" / "pre"
IMAGES_OUTPUT_PATH = Path(__file__).parent / "static" / "images" / "output"
MODEL_PATH = Path(__file__).parent / "models" / "modello_del_peter.onnx"
METADATA_PATH = IMAGES_PRE_PATH / "metadata.json"

# Import inference functions
sys.path.insert(0, str(Path(__file__).parent))
from inference import process_image_batch, load_metadata


# ============================================================================
# HELPER FUNCTIONS - All defined at top before main code
# ============================================================================


@st.cache_data
def load_geojson():
    """Load GeoJSON data from file."""
    if os.path.exists(GEOJSON_PATH):
        with open(GEOJSON_PATH, "r") as f:
            return json.load(f)
    return None


def add_geojson_markers(map_obj, geojson_data, label_filter=None):
    """Add markers from GeoJSON to the map."""
    if not geojson_data:
        return

    # Label mapping
    LABEL_MAP = {1: "Crack", 2: "Manhole", 3: "Pothole"}

    for feature in geojson_data.get("features", []):
        coords = feature["geometry"]["coordinates"]
        props = feature["properties"]

        # GeoJSON uses [longitude, latitude]
        lat, lon = coords[1], coords[0]

        # Create abbreviated coordinate label for marker
        marker_label = f"{lat:.3f}, {lon:.3f}"

        # Extract image info
        image_filename = props.get("image", "")
        image_id = props.get("image_id", "Unknown")

        # Extract and process all labels
        labels = props.get("labels", [])

        # Filter markers based on label_filter
        if label_filter is not None:
            # Check if any of the marker's labels match the filter
            marker_labels = {label_item.get("label") for label_item in labels}
            if not any(label_id in label_filter for label_id in marker_labels):
                continue  # Skip this marker

        detections_html = ""

        if labels and len(labels) > 0:
            detections_html = (
                "<div style='margin-top: 10px; padding-top: 8px; border-top: 1px solid #ddd;'>"
            )
            detections_html += "<strong style='font-size: 13px;'>Detections:</strong><br>"

            for label_item in labels:
                label_id = label_item.get("label")
                label_name = LABEL_MAP.get(label_id, f"Unknown ({label_id})")
                avg_conf = label_item.get("avg_confidence", 0)
                count = label_item.get("count", 1)

                # Color code by label type
                color = "#e74c3c" if label_id == 1 else "#3498db" if label_id == 2 else "#e67e22"

                detections_html += f"""
                <div style='margin: 5px 0; padding: 5px; background-color: {color}15; border-left: 3px solid {color}; border-radius: 3px;'>
                    <span style='font-weight: bold; color: {color};'>{label_name}</span><br>
                    <span style='font-size: 11px;'>Confidence: {avg_conf:.1%}</span>
                    {f"<span style='font-size: 11px;'> | Count: {count}</span>" if count > 1 else ""}
                </div>
                """

            detections_html += "</div>"

        # Load and encode detected image as base64
        images_html = ""
        if image_filename:
            # Detected image path (static/images/output folder)
            detected_filename = f"{Path(image_filename).stem}_detected{Path(image_filename).suffix}"
            detected_path = (
                Path(__file__).parent / "static" / "images" / "output" / detected_filename
            )

            if detected_path.exists():
                try:
                    with open(detected_path, "rb") as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                        images_html = f'<img src="data:image/jpeg;base64,{img_data}" style="width: 100%; height: auto; border-radius: 4px; margin-bottom: 8px;">'
                except Exception:
                    pass

        # Create popup with image and info
        popup_html = f"""
        <div style="width: 280px; font-family: Arial, sans-serif;">
            <h4 style="margin: 0 0 10px 0; color: #2c3e50;">📍 {marker_label}</h4>
            {images_html}
            <p style="margin: 5px 0; font-size: 12px; color: #7f8c8d;"><strong>Image ID:</strong> {image_id}</p>
            {detections_html}
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=marker_label,
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(map_obj)


def create_map(location, zoom=12, show_geojson=True, label_filter=None, use_detections=False):
    """Create a folium map centered on location."""
    m = folium.Map(location=location, zoom_start=zoom, tiles="OpenStreetMap")

    # Add click event to get coordinates - restore original LatLngPopup
    m.add_child(folium.LatLngPopup())

    # Add GeoJSON markers
    if show_geojson:
        if use_detections:
            # Load from detections.json (Mapillary mode after inference)
            detections_file = IMAGES_OUTPUT_PATH / "detections.json"
            if detections_file.exists():
                with open(detections_file, "r") as f:
                    geojson_data = json.load(f)
            else:
                geojson_data = None
        else:
            # Load from points.geojson (pre-loaded mode)
            geojson_data = load_geojson()
        
        add_geojson_markers(m, geojson_data, label_filter)

    return m


def download_mapillary_images(lat, lon, radius_km, output_dir="mapillary_downloads"):
    """
    Download Mapillary images for the specified area.

    Args:
        lat: Latitude
        lon: Longitude
        radius_km: Radius in kilometers
        output_dir: Output directory for images

    Returns:
        Path to downloaded images
    """
    try:
        # Clear previous images
        if IMAGES_PRE_PATH.exists():
            for f in IMAGES_PRE_PATH.glob("*"):
                if f.is_file():
                    f.unlink()
        else:
            IMAGES_PRE_PATH.mkdir(parents=True, exist_ok=True)

        # Run download command
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "download_mapillary.py"),
            str(lat),
            str(lon),
            str(radius_km),
            "--output",
            str(IMAGES_PRE_PATH.parent / "pre_temp"),  # Use temp directory
        ]

        if MAPILLARY_API_KEY != "YOUR_API_KEY_HERE":
            cmd.extend(["--api-key", MAPILLARY_API_KEY])

        # Create placeholders for progress feedback
        progress_container = st.container()
        status_text = progress_container.empty()
        progress_bar = progress_container.progress(0)

        # Run with real-time progress output (unbuffered)
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            bufsize=1,  # Line buffered
            env=env
        )

        fetch_count = 0
        download_current = 0
        download_total = 0

        # Read output in real-time
        try:
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break

                line = line.strip()
                
                # Debug: log all output
                print(f"[DEBUG] {line}", file=sys.stderr)

                # Parse progress output
                if line.startswith("PROGRESS:"):
                    try:
                        fetch_count = int(line.split(":")[1])
                        status_text.text(f"📥 Fetching metadata: {fetch_count} images found...")
                        progress_bar.progress(min(0.3, fetch_count / 10000))
                    except (ValueError, IndexError):
                        pass

                elif line.startswith("DOWNLOAD_PROGRESS:"):
                    try:
                        parts = line.split(":")[1].split("/")
                        download_current = int(parts[0])
                        download_total = int(parts[1])
                        pct = min(0.3 + 0.7 * (download_current / download_total), 0.99)
                        status_text.text(
                            f"⬇️ Downloading: {download_current}/{download_total} images"
                        )
                        progress_bar.progress(pct)
                    except (ValueError, IndexError):
                        pass
        except Exception as e:
            st.warning(f"Progress tracking error: {e}")

        process.wait()

        if process.returncode != 0:
            st.error(f"Download error occurred")
            return None

        status_text.text("🔄 Processing downloaded files...")
        progress_bar.progress(0.95)

        # Find the actual created directory (with timestamp)
        parent_dir = IMAGES_PRE_PATH.parent
        temp_dirs = list(parent_dir.glob("pre_temp_*"))
        
        if not temp_dirs:
            st.error("Download completed but output directory not found")
            return None
        
        # Use the most recent one
        actual_download_dir = sorted(temp_dirs)[-1]
        
        # Move files from temp directory to pre directory
        for f in actual_download_dir.glob("*"):
            if f.is_file():
                f.rename(IMAGES_PRE_PATH / f.name)
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(actual_download_dir, ignore_errors=True)

        progress_bar.progress(1.0)
        status_text.text(f"✅ Downloaded {download_total} images successfully!")

        return IMAGES_PRE_PATH

    except subprocess.TimeoutExpired:
        st.error("Download timeout - too many images or slow connection")
        return None
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        return None


def run_inference_on_images(confidence_threshold=0.74):
    """
    Run inference on all images in the pre directory.

    Args:
        confidence_threshold: Confidence threshold for detections

    Returns:
        Results dictionary or None if error
    """
    try:
        # Check if images exist
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = [
            f
            for f in IMAGES_PRE_PATH.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            st.warning("No images found in pre directory")
            return None

        # Create output directory
        IMAGES_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

        # Load metadata
        metadata_map = {}
        if METADATA_PATH.exists():
            metadata_map = load_metadata(str(METADATA_PATH))

        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Process images
        status_text.text(f"Processing {len(image_files)} images...")
        results = process_image_batch(
            str(MODEL_PATH),
            [str(f) for f in image_files],
            metadata_map=metadata_map,
            output_dir=str(IMAGES_OUTPUT_PATH),
            output_json_path=str(IMAGES_OUTPUT_PATH / "detections.json"),
            confidence_threshold=confidence_threshold,
            verbose=False,
        )

        # Update progress
        progress_bar.progress(100)
        status_text.text("✅ Inference complete!")

        return results

    except Exception as e:
        st.error(f"Inference error: {str(e)}")
        return None


# ============================================================================
# MAIN APP
# ============================================================================

st.title("🗺️ City Map Dashboard with Inference")

# Store marker mode in session state to access it later
if "marker_mode" not in st.session_state:
    st.session_state.marker_mode = "Local GeoJSON"

if "inference_complete" not in st.session_state:
    st.session_state.inference_complete = False

if "mapillary_processed" not in st.session_state:
    st.session_state.mapillary_processed = False

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Mode selection
    st.markdown("### 📍 Data Source")
    st.session_state.marker_mode = st.radio(
        "Choose mode:",
        ["Local GeoJSON", "Mapillary API"],
        help="Local GeoJSON: Use pre-loaded dataset. Mapillary API: Download new images from map.",
    )
    marker_mode = st.session_state.marker_mode

    st.markdown("---")

    # Inference settings
    st.markdown("### 🤖 Inference Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.74, step=0.01
    )

    st.markdown("---")

    if marker_mode == "Mapillary API":
        if MAPILLARY_API_KEY == "YOUR_API_KEY_HERE":
            st.warning("⚠️ Mapillary API key not configured!")
            st.info("Add your API key to `.streamlit/secrets.toml`")
            st.code('MAPILLARY_API_KEY = "your_key_here"', language="toml")
            st.markdown("[Get a free API key →](https://www.mapillary.com/developer)")
        else:
            st.success("✅ Mapillary API key configured")

        search_radius_km = st.slider(
            "Download radius (km)", min_value=0.1, max_value=5.0, value=1.0, step=0.1
        )
    else:
        search_radius_km = 1.0
        geojson_data = load_geojson()
        if geojson_data:
            num_markers = len(geojson_data.get("features", []))
            st.info(f"📌 {num_markers} markers loaded from GeoJSON")

            # Label filter checkboxes
            st.markdown("### 🔍 Filter by Detection Type")
            selected_labels = st.multiselect(
                "Show markers containing:",
                options=[1, 2, 3],
                format_func=lambda x: {1: "🔴 Crack", 2: "🔵 Manhole", 3: "🟠 Pothole"}[x],
                default=[1, 2, 3],
                help="Select which types of detections to display on the map",
            )

            # Store in session state
            st.session_state.label_filter = selected_labels if selected_labels else None
        else:
            st.warning("⚠️ No GeoJSON file found")
            st.session_state.label_filter = None

    st.markdown("---")
    
    st.markdown("### 📖 How to Use")
    if marker_mode == "Mapillary API":
        st.markdown("""
        1. Enter a city name
        2. Click on map to place a pin
        3. Click 'Download & Process'
        4. Wait for images to download and inference to run
        5. View results on map
        """)
    else:
        st.markdown("""
        1. Enter a city name (optional)
        2. View detections from pre-loaded dataset
        3. Click 'Run Inference' to reprocess images
        4. Click markers for details
        """)

# City search - left-aligned with form for Enter key support
st.markdown("### 🔍 Search City")
with st.form("search_form"):
    col1, col2 = st.columns([2, 1])
    with col1:
        city = st.text_input("Enter city name", placeholder="e.g., Paris, Tokyo, New York", label_visibility="collapsed")
    with col2:
        search_button = st.form_submit_button("🔍 Search", use_container_width=True)

# Initialize session state
if "map_location" not in st.session_state:
    st.session_state.map_location = [48.8566, 2.3522]  # Default: Paris
    st.session_state.zoom = 12
    st.session_state.city_name = "Paris"

# Handle city search
if search_button and city:
    with st.spinner(f"Searching for {city}..."):
        try:
            geolocator = Nominatim(user_agent="streamlit_map_app")
            location = geolocator.geocode(city, timeout=10)

            if location:
                st.session_state.map_location = [location.latitude, location.longitude]
                st.session_state.zoom = 12
                st.session_state.city_name = city
                st.success(f"✓ Found: {location.address}")
            else:
                st.error(f"Could not find location: {city}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Create and display map
mode_indicator = (
    "🗺️ GeoJSON Markers" if st.session_state.marker_mode == "Local GeoJSON" else "📸 Mapillary Mode"
)
st.subheader(f"Map: {st.session_state.city_name} | {mode_indicator}")

# Determine if we should show detections (Mapillary mode with processed results)
use_detections = st.session_state.marker_mode == "Mapillary API" and st.session_state.get("mapillary_processed", False)

# Create map with or without GeoJSON markers based on mode
show_geojson = st.session_state.marker_mode == "Local GeoJSON" or use_detections
label_filter = st.session_state.get("label_filter", None) if st.session_state.marker_mode == "Local GeoJSON" else None
map_obj = create_map(
    st.session_state.map_location, st.session_state.zoom, show_geojson, label_filter, use_detections
)

# Display map and capture clicks
map_data = st_folium(map_obj, width=None, height=600, returned_objects=["last_clicked"])

# Handle map clicks (Mapillary mode) - show download button below map
if st.session_state.marker_mode == "Mapillary API":
    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]

        # Store clicked location in session state
        st.session_state.clicked_lat = clicked_lat
        st.session_state.clicked_lon = clicked_lon

        st.markdown("---")
        # Download & Process button - appears below the map
        if st.button("⬇️ Download & Process", use_container_width=True, key="download_process"):
            with st.spinner("Downloading images from Mapillary..."):
                download_result = download_mapillary_images(
                    clicked_lat, clicked_lon, search_radius_km
                )

                if download_result:
                    st.success(f"✅ Images downloaded to {download_result}")

                    # Run inference
                    with st.spinner("Running inference..."):
                        results = run_inference_on_images(confidence_threshold)
                        if results:
                            st.session_state.inference_complete = True
                            st.session_state.mapillary_processed = True  # Flag for map refresh
                            st.success("✅ Inference complete!")

                            # Show results
                            successful = sum(
                                1
                                for r in results
                                if "boxes" in r and len(r["boxes"]) > 0
                            )
                            st.info(
                                f"Processed {len(results)} images, {successful} with detections"
                            )
                            
                            # Force rerun to show updated map with detections
                            st.rerun()

elif st.session_state.marker_mode == "Local GeoJSON":
    st.markdown("---")

    # Run inference button for pre-loaded dataset
    if st.button("🚀 Run Inference on Pre-loaded Images", use_container_width=True):
        with st.spinner("Running inference on pre-loaded images..."):
            results = run_inference_on_images(confidence_threshold)
            if results:
                st.session_state.inference_complete = True
                successful = sum(1 for r in results if "boxes" in r and len(r["boxes"]) > 0)
                st.success(f"✅ Processed {len(results)} images, {successful} with detections")

    st.markdown("""
    🗺️ **GeoJSON Mode**: 
    - Red markers show detected road issues from your local database
    - Click markers for details and detection images
    - Use the 'Run Inference' button to reprocess images
    """)

# Display detections summary if inference was run
if st.session_state.inference_complete:
    detections_file = IMAGES_OUTPUT_PATH / "detections.json"
    if detections_file.exists():
        st.markdown("---")
        st.subheader("📊 Detection Summary")

        with open(detections_file, "r") as f:
            data = json.load(f)

        if "summary" in data:
            summary = data["summary"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Images", summary.get("total_images", 0))
            col2.metric("Total Detections", summary.get("total_detections", 0))
            col3.metric("Avg Confidence", f"{summary['confidence_stats'].get('avg', 0):.2%}")
            col4.metric(
                "Labels Found",
                len(summary.get("unique_labels", [])),
            )

# Footer
st.markdown("---")
footer_text = (
    "Data sources: OpenStreetMap, Mapillary"
    if st.session_state.marker_mode == "Mapillary API"
    else "Data sources: OpenStreetMap, Local GeoJSON"
)
st.caption(footer_text)
