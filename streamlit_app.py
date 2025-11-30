import base64
import json
import os
import shutil
import subprocess
import sys
import time
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
IMAGES_PRE_USERS_PATH = Path(__file__).parent / "static" / "images" / "pre_users"
IMAGES_OUTPUT_PATH = Path(__file__).parent / "static" / "images" / "output"
MODEL_PATH = Path(__file__).parent / "models" / "modello_del_peter.onnx"
METADATA_PATH = IMAGES_PRE_PATH / "metadata.json"
METADATA_USERS_PATH = IMAGES_PRE_USERS_PATH / "metadata.json"

# Import inference functions
sys.path.insert(0, str(Path(__file__).parent))
from inference import load_metadata, process_image_batch

# ============================================================================
# HELPER FUNCTIONS - All defined at top before main code
# ============================================================================


def load_geojson():
    """Load GeoJSON data from file, including user-uploaded metadata.
    NOT cached because we need fresh data when new images are uploaded."""
    features = []

    # Load pre-loaded GeoJSON
    if os.path.exists(GEOJSON_PATH):
        with open(GEOJSON_PATH, "r") as f:
            geojson_data = json.load(f)
            features.extend(geojson_data.get("features", []))

    # Load user-uploaded metadata and convert to GeoJSON features
    if METADATA_USERS_PATH.exists():
        try:
            with open(METADATA_USERS_PATH, "r") as f:
                metadata_list = json.load(f)
                for entry in metadata_list:
                    feature = {
                        "type": "Feature",
                        "geometry": entry.get("geometry", {"type": "Point", "coordinates": [0, 0]}),
                        "properties": {
                            "image": entry.get("image", ""),
                            "image_id": entry.get("image_id", ""),
                            "captured_at": entry.get("captured_at", ""),
                            "labels": entry.get("labels", []),
                        },
                    }
                    features.append(feature)
        except Exception as e:
            pass  # Silently fail if metadata file is empty or malformed

    if features:
        return {"type": "FeatureCollection", "features": features}
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

        # Determine dominant label (highest count)
        dominant_label = None
        max_count = 0
        if labels and len(labels) > 0:
            for label_item in labels:
                count = label_item.get("count", 1)
                if count > max_count:
                    max_count = count
                    dominant_label = label_item.get("label")

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
                        # Enlarged image styling
                        images_html = f'<img src="data:image/jpeg;base64,{img_data}" style="width: 100%; height: auto; max-height: 500px; border-radius: 8px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);">'
                except Exception:
                    pass

        # Create popup with image and info (enlarged popup)
        popup_html = f"""
        <div style="width: 450px; font-family: Arial, sans-serif; padding: 10px;">
            <h4 style="margin: 0 0 10px 0; color: #2c3e50; font-size: 16px;">📍 {marker_label}</h4>
            {images_html}
            <p style="margin: 8px 0; font-size: 12px; color: #7f8c8d;"><strong>Image ID:</strong> {image_id}</p>
            {detections_html}
        </div>
        """

        # Determine marker color based on dominant label
        marker_color = "gray"  # Default color
        if dominant_label == 1:
            marker_color = "red"  # Cracks
        elif dominant_label == 2:
            marker_color = "blue"  # Manholes
        elif dominant_label == 3:
            marker_color = "orange"  # Potholes

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=500),
            tooltip=marker_label,
            icon=folium.Icon(color=marker_color, icon="info-sign"),
        ).add_to(map_obj)


def create_map(location, zoom=13, show_geojson=True, label_filter=None, use_detections=False):
    """Create a folium map centered on location."""
    m = folium.Map(location=location, zoom_start=zoom, tiles="OpenStreetMap")

    # Add click event to get coordinates - restore original LatLngPopup
    m.add_child(folium.LatLngPopup())

    # Add GeoJSON markers
    if show_geojson:
        # Prefer detections.json if it exists (inference has been run)
        detections_file = IMAGES_OUTPUT_PATH / "detections.json"
        if detections_file.exists():
            # Load from detections.json (after inference, regardless of mode)
            with open(detections_file, "r") as f:
                geojson_data = json.load(f)
        else:
            # Load from points.geojson or user metadata (before inference)
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
        env["PYTHONUNBUFFERED"] = "1"

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            env=env,
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
        # Check if images exist in both pre and pre_users directories
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = []

        # Collect images from pre directory
        if IMAGES_PRE_PATH.exists():
            image_files.extend(
                [
                    f
                    for f in IMAGES_PRE_PATH.iterdir()
                    if f.is_file() and f.suffix.lower() in image_extensions
                ]
            )

        # Collect images from pre_users directory
        if IMAGES_PRE_USERS_PATH.exists():
            image_files.extend(
                [
                    f
                    for f in IMAGES_PRE_USERS_PATH.iterdir()
                    if f.is_file() and f.suffix.lower() in image_extensions
                ]
            )

        if not image_files:
            st.warning("No images found in pre or pre_users directories")
            return None

        # Create output directory
        IMAGES_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

        # Load metadata from both sources
        metadata_map = {}
        if METADATA_PATH.exists():
            metadata_map.update(load_metadata(str(METADATA_PATH)))
        if METADATA_USERS_PATH.exists():
            metadata_map.update(load_metadata(str(METADATA_USERS_PATH)))

        # CLEAN output directory THOROUGHLY before inference
        print(f"Attempting to clean output directory: {IMAGES_OUTPUT_PATH}")
        try:
            if IMAGES_OUTPUT_PATH.exists():
                # Use shutil.rmtree to forcefully remove entire directory
                shutil.rmtree(IMAGES_OUTPUT_PATH)
                print(f"Successfully removed entire output directory")
            # Recreate fresh directory
            IMAGES_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
            print(f"Created fresh output directory: {IMAGES_OUTPUT_PATH}")
        except Exception as e:
            print(f"Error cleaning output directory: {e}")
            # Fallback: try to delete files individually
            if IMAGES_OUTPUT_PATH.exists():
                for f in list(IMAGES_OUTPUT_PATH.iterdir()):
                    if f.is_file():
                        try:
                            f.unlink()
                            print(f"Deleted file: {f}")
                        except Exception as e2:
                            print(f"Could not delete {f}: {e2}")

        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Show cleanup status
        status_text.text(
            f"Output cleaned. Processing {len(image_files)} images from both sources..."
        )
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
    # Mode selection
    st.markdown("**📍 Data Source**")
    st.session_state.marker_mode = st.radio(
        "Choose mode:",
        ["Local GeoJSON", "Mapillary API"],
        help="Local GeoJSON: Use pre-loaded dataset. Mapillary API: Download new images from map.",
        label_visibility="collapsed",
    )
    marker_mode = st.session_state.marker_mode

    # Inference settings
    st.markdown("**🤖 Confidence Threshold**")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.74,
        step=0.01,
        label_visibility="collapsed",
    )

    if marker_mode == "Mapillary API":
        if MAPILLARY_API_KEY == "YOUR_API_KEY_HERE":
            st.warning("⚠️ API key not configured")
        else:
            st.success("✅ API key OK")

        st.markdown("**📏 Download Radius**")
        search_radius_km = st.slider(
            "Download radius (km)",
            min_value=0.1,
            max_value=5.0,
            value=0.2,
            step=0.1,
            label_visibility="collapsed",
        )
    else:
        search_radius_km = 0.2
        geojson_data = load_geojson()
        if geojson_data:
            num_markers = len(geojson_data.get("features", []))
            st.caption(f"📌 {num_markers} markers loaded")

            # Label filter checkboxes
            st.markdown("**🔍 Filter by Type**")
            selected_labels = st.multiselect(
                "Show markers:",
                options=[1, 2, 3],
                format_func=lambda x: {1: "🔴 Crack", 2: "🔵 Manhole", 3: "🟠 Pothole"}[x],
                default=[1, 2, 3],
                label_visibility="collapsed",
            )

            # Store in session state
            st.session_state.label_filter = selected_labels if selected_labels else None
        else:
            st.warning("⚠️ No GeoJSON file found")
            st.session_state.label_filter = None

    # Display results summary in sidebar
    st.markdown("---")
    st.markdown("**📊 Results**")

    # Determine which data source to use
    data_to_analyze = None

    if marker_mode == "Mapillary API" and st.session_state.inference_complete:
        # Use detections.json for Mapillary mode after inference
        detections_file = IMAGES_OUTPUT_PATH / "detections.json"
        if detections_file.exists():
            with open(detections_file, "r") as f:
                data_to_analyze = json.load(f)
    elif marker_mode == "Local GeoJSON":
        # Use points.geojson for Local GeoJSON mode
        data_to_analyze = load_geojson()

    # Calculate and display statistics
    if data_to_analyze and "features" in data_to_analyze:
        total_images = len(data_to_analyze.get("features", []))
        total_detections = 0
        label_counts = {1: 0, 2: 0, 3: 0}

        for feature in data_to_analyze.get("features", []):
            labels = feature.get("properties", {}).get("labels", [])
            for label_item in labels:
                label_id = label_item.get("label")
                count = label_item.get("count", 1)
                total_detections += count
                if label_id in label_counts:
                    label_counts[label_id] += count

        st.markdown(f"**Images:** **{total_images}**")
        st.markdown(f"**Total Detections:** **{total_detections}**")

        st.markdown("**By Type:**")
        if label_counts[1] > 0:
            st.markdown(f"🔴 **Cracks:** **{label_counts[1]}**")
        if label_counts[2] > 0:
            st.markdown(f"🔵 **Manholes:** **{label_counts[2]}**")
        if label_counts[3] > 0:
            st.markdown(f"🟠 **Potholes:** **{label_counts[3]}**")
    else:
        st.caption("No data to display")

# City search - left-aligned with form for Enter key support
st.markdown("### 🔍 Search City")
with st.form("search_form"):
    col1, col2 = st.columns([2, 1])
    with col1:
        city = st.text_input(
            "Enter city name",
            placeholder="e.g., Paris, Tokyo, New York",
            label_visibility="collapsed",
        )
    with col2:
        search_button = st.form_submit_button("🔍 Search", use_container_width=True)

# Initialize session state
if "map_location" not in st.session_state:
    st.session_state.map_location = [48.1351, 11.5820]  # Default: Munich
    st.session_state.zoom = 13
    st.session_state.city_name = "Munich"
    st.session_state.marker_mode = "Local GeoJSON"  # Default mode

if "marker_mode" not in st.session_state:
    st.session_state.marker_mode = "Local GeoJSON"

# Handle city search
if search_button and city:
    with st.spinner(f"Searching for {city}..."):
        try:
            geolocator = Nominatim(user_agent="streamlit_map_app")
            location = geolocator.geocode(city, timeout=10)

            if location:
                st.session_state.map_location = [location.latitude, location.longitude]
                st.session_state.zoom = 13
                st.session_state.city_name = city
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
use_detections = st.session_state.marker_mode == "Mapillary API" and st.session_state.get(
    "mapillary_processed", False
)

# Create map with or without GeoJSON markers based on mode
show_geojson = st.session_state.marker_mode == "Local GeoJSON" or use_detections
label_filter = (
    st.session_state.get("label_filter", None)
    if st.session_state.marker_mode == "Local GeoJSON"
    else None
)
map_obj = create_map(
    st.session_state.map_location, st.session_state.zoom, show_geojson, label_filter, use_detections
)

# Force map to refresh by using a key based on detections file modification time
# This ensures the map updates when new detections are available
detections_file = IMAGES_OUTPUT_PATH / "detections.json"
map_key = "map_base"
if detections_file.exists():
    try:
        # Use file modification time to create unique key
        mtime = detections_file.stat().st_mtime
        map_key = f"map_{int(mtime * 1000)}"
    except:
        pass

# Display map and capture clicks
map_data = st_folium(
    map_obj, width=None, height=600, returned_objects=["last_clicked"], key=map_key
)

# Handle map clicks (Mapillary mode) - show modal dialog
if st.session_state.marker_mode == "Mapillary API":
    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]

        # Store clicked location in session state
        st.session_state.clicked_lat = clicked_lat
        st.session_state.clicked_lon = clicked_lon
        st.session_state.show_download_modal = True

# Show modal dialog when location is clicked
if st.session_state.marker_mode == "Mapillary API" and st.session_state.get(
    "show_download_modal", False
):

    @st.dialog("📍 Download Mapillary Images")
    def download_modal():
        clicked_lat = st.session_state.get("clicked_lat")
        clicked_lon = st.session_state.get("clicked_lon")

        st.write(f"**Location:** {clicked_lat:.6f}, {clicked_lon:.6f}")
        st.write(f"**Radius:** {search_radius_km} km")
        st.write(f"**Confidence threshold:** {confidence_threshold:.0%}")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            download_button = st.button(
                "⬇️ Download & Process", use_container_width=True, type="primary"
            )

        with col2:
            cancel_button = st.button("❌ Cancel", use_container_width=True)

        # Handle button actions outside columns so progress bars span full width
        if download_button:
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
                            st.session_state.mapillary_processed = True
                            st.success("✅ Inference complete!")

                            # Show results
                            successful = sum(
                                1 for r in results if "boxes" in r and len(r["boxes"]) > 0
                            )
                            st.info(
                                f"Processed {len(results)} images, {successful} with detections"
                            )

                            # Close modal and force rerun
                            st.session_state.show_download_modal = False
                            st.rerun()

        if cancel_button:
            st.session_state.show_download_modal = False
            st.rerun()

    download_modal()

elif st.session_state.marker_mode == "Local GeoJSON":
    st.markdown("---")

    # Check if we need to run inference:
    # 1. First time in session (preload_inference_done not set)
    # 2. New images in pre_users directory (crowdsourced uploads)
    def should_run_inference():
        if not st.session_state.get("preload_inference_done", False):
            return True

        # Check if there are new unprocessed images in pre_users
        if IMAGES_PRE_USERS_PATH.exists():
            user_images = (
                list(IMAGES_PRE_USERS_PATH.glob("*.jpg"))
                + list(IMAGES_PRE_USERS_PATH.glob("*.jpeg"))
                + list(IMAGES_PRE_USERS_PATH.glob("*.png"))
            )
            if user_images:
                # Check if any user images don't have a corresponding detected version
                for img_path in user_images:
                    detected_filename = f"{img_path.stem}_detected{img_path.suffix}"
                    detected_path = IMAGES_OUTPUT_PATH / detected_filename
                    if not detected_path.exists():
                        return True  # Found unprocessed image
        return False

    # Auto-run inference if needed
    if should_run_inference():
        with st.spinner("Running inference on pre-loaded and user-uploaded images..."):
            results = run_inference_on_images(confidence_threshold)

            # Clear cache after inference
            st.cache_data.clear()
            if results:
                st.session_state.inference_complete = True
                st.session_state.preload_inference_done = True
                successful = sum(1 for r in results if "boxes" in r and len(r["boxes"]) > 0)
                st.success(f"✅ Processed {len(results)} images, {successful} with detections")
                st.rerun()
            else:
                st.session_state.preload_inference_done = True

    st.markdown("""
    🗺️ **GeoJSON Mode**: 
    - Red markers show detected road issues from your local database
    - Click markers for details and detection images with rectangles
    - Inference runs automatically on all pre-loaded and user-uploaded images
    """)

# Footer
st.markdown("---")
footer_text = (
    "Data sources: OpenStreetMap, Mapillary"
    if st.session_state.marker_mode == "Mapillary API"
    else "Data sources: OpenStreetMap, Local GeoJSON"
)
st.caption(footer_text)

# Auto-refresh when new uploads are detected
if st.session_state.marker_mode == "Local GeoJSON":
    # Check for upload signal file (created by mobile_upload.py)
    SIGNAL_FILE = Path(__file__).parent / ".upload_signal"
    if SIGNAL_FILE.exists():
        try:
            SIGNAL_FILE.unlink()  # Remove signal file
            st.rerun()  # Trigger immediate refresh
        except:
            pass
    
    # Also track metadata file modification as fallback
    if "last_metadata_mtime" not in st.session_state:
        if METADATA_USERS_PATH.exists():
            st.session_state.last_metadata_mtime = METADATA_USERS_PATH.stat().st_mtime
        else:
            st.session_state.last_metadata_mtime = 0

    # Check if metadata file has been modified
    current_mtime = METADATA_USERS_PATH.stat().st_mtime if METADATA_USERS_PATH.exists() else 0

    if current_mtime != st.session_state.last_metadata_mtime:
        st.session_state.last_metadata_mtime = current_mtime
        st.rerun()

    # Manual refresh button
    if st.button("🔄 Check for New Uploads", use_container_width=True):
        st.rerun()
    
    # Auto-refresh every 5 seconds when in Local GeoJSON mode
    time.sleep(5)
    st.rerun()
