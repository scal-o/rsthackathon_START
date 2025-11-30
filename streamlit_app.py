import json
import os

import folium
import requests
import streamlit as st
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium

# Page config
st.set_page_config(page_title="City Map Dashboard with Mapillary", page_icon="🗺️", layout="wide")

# Mapillary API configuration
MAPILLARY_API_KEY = st.secrets.get("MAPILLARY_API_KEY", "YOUR_API_KEY_HERE")
MAPILLARY_API_URL = "https://graph.mapillary.com/images"

# Paths
GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "data", "points.geojson")


@st.cache_data
def load_geojson():
    """Load GeoJSON data from file."""
    if os.path.exists(GEOJSON_PATH):
        with open(GEOJSON_PATH, "r") as f:
            return json.load(f)
    return None


def get_mapillary_image(lat, lon, radius=50):
    """
    Get the closest Mapillary image near the given coordinates.

    Args:
        lat: Latitude
        lon: Longitude
        radius: Search radius in meters (default 50)

    Returns:
        Dictionary with image info or None
    """
    try:
        params = {
            "access_token": MAPILLARY_API_KEY,
            "fields": "id,thumb_1024_url,captured_at,compass_angle",
            "bbox": f"{lon - 0.001},{lat - 0.001},{lon + 0.001},{lat + 0.001}",
            "limit": 1,
        }

        response = requests.get(MAPILLARY_API_URL, params=params, timeout=5)

        if response.status_code == 200:
            data = response.json()
            if data.get("data") and len(data["data"]) > 0:
                return data["data"][0]
        return None
    except Exception as e:
        st.error(f"Mapillary API error: {str(e)}")
        return None


def add_geojson_markers(map_obj, geojson_data, label_filter=None):
    """Add markers from GeoJSON to the map."""
    if not geojson_data:
        return

    import base64
    from pathlib import Path

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


def create_map(location, zoom=12, show_geojson=True, label_filter=None):
    """Create a folium map centered on location."""
    m = folium.Map(location=location, zoom_start=zoom, tiles="OpenStreetMap")

    # Add click event to get coordinates
    m.add_child(folium.LatLngPopup())

    # Add GeoJSON markers only if enabled
    if show_geojson:
        geojson_data = load_geojson()
        add_geojson_markers(m, geojson_data, label_filter)

    return m


# Main app
st.title("🗺️ City Map Dashboard")

# Store marker mode in session state to access it later
if "marker_mode" not in st.session_state:
    st.session_state.marker_mode = "Local GeoJSON"

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Mode selection
    st.markdown("### 📍 Marker Mode")
    st.session_state.marker_mode = st.radio(
        "Choose data source:",
        ["Local GeoJSON", "Mapillary API"],
        help="Local GeoJSON shows predefined markers. Mapillary API allows clicking to load street view images.",
    )
    marker_mode = st.session_state.marker_mode

    st.markdown("---")

    if marker_mode == "Mapillary API":
        if MAPILLARY_API_KEY == "YOUR_API_KEY_HERE":
            st.warning("⚠️ Mapillary API key not configured!")
            st.info("Add your API key to `.streamlit/secrets.toml`")
            st.code('MAPILLARY_API_KEY = "your_key_here"', language="toml")
            st.markdown("[Get a free API key →](https://www.mapillary.com/developer)")
        else:
            st.success("✅ Mapillary API key configured")

        search_radius = st.slider("Search radius (meters)", 10, 200, 50)
    else:
        search_radius = 50  # Default value when not using Mapillary
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
    st.markdown("### How to use")
    if marker_mode == "Mapillary API":
        st.markdown("""
        1. Enter a city name below
        2. Click anywhere on the map
        3. Click 'Fetch Mapillary Image'
        4. View street view photos
        """)
    else:
        st.markdown("""
        1. Enter a city name below
        2. View predefined markers on map
        3. Click markers for info
        """)

# City search
col1, col2 = st.columns([3, 1])
with col1:
    city = st.text_input("Enter city name", placeholder="e.g., Paris, Tokyo, New York")
with col2:
    search_button = st.button("🔍 Search", use_container_width=True)

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

# Create map with or without GeoJSON markers based on mode
show_geojson = st.session_state.marker_mode == "Local GeoJSON"
label_filter = st.session_state.get("label_filter", None) if show_geojson else None
map_obj = create_map(
    st.session_state.map_location, st.session_state.zoom, show_geojson, label_filter
)

# Display map and capture clicks
map_data = st_folium(map_obj, width=None, height=600, returned_objects=["last_clicked"])

# Handle map clicks (only in Mapillary mode)
if st.session_state.marker_mode == "Mapillary API":
    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]

        st.markdown("---")
        st.subheader("📍 Clicked Location")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Latitude", f"{clicked_lat:.6f}")
            st.metric("Longitude", f"{clicked_lon:.6f}")

            if st.button("🔄 Fetch Mapillary Image"):
                with st.spinner("Searching for nearby Mapillary images..."):
                    image_data = get_mapillary_image(clicked_lat, clicked_lon, search_radius)

                    if image_data:
                        st.session_state.mapillary_image = image_data
                    else:
                        st.warning(
                            "No Mapillary images found nearby. Try increasing the search radius."
                        )
                        st.session_state.mapillary_image = None

        with col2:
            if "mapillary_image" in st.session_state and st.session_state.mapillary_image:
                img_data = st.session_state.mapillary_image
                st.image(
                    img_data["thumb_1024_url"],
                    caption=f"Captured: {img_data.get('captured_at', 'Unknown date')}",
                    use_container_width=True,
                )

                # Link to full image on Mapillary
                mapillary_url = f"https://www.mapillary.com/app/?pKey={img_data['id']}"
                st.markdown(f"[View on Mapillary →]({mapillary_url})")
            else:
                st.info("👆 Click 'Fetch Mapillary Image' to load street view")
elif st.session_state.marker_mode == "Local GeoJSON":
    st.markdown("---")
    st.info(
        "🗺️ **GeoJSON Mode**: Red markers on the map show locations from your local database. Click on any marker to see details."
    )

# Footer
st.markdown("---")
if st.session_state.marker_mode == "Mapillary API":
    st.caption("Data sources: OpenStreetMap, Mapillary")
else:
    st.caption("Data sources: OpenStreetMap, Local GeoJSON")
