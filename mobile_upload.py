#!/usr/bin/env python3
"""
Mobile-friendly pothole upload interface.
Uploads images to the pre directory with geolocation from browser GPS.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit_geolocation import streamlit_geolocation

# Page config
st.set_page_config(
    page_title="Pothole Reporter",
    page_icon="🕳️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Paths
IMAGES_PRE_USERS_PATH = Path(__file__).parent / "static" / "images" / "pre_users"
METADATA_PATH = IMAGES_PRE_USERS_PATH / "metadata.json"

# Create directories if they don't exist
IMAGES_PRE_USERS_PATH.mkdir(parents=True, exist_ok=True)


def load_metadata():
    """Load existing metadata or create empty dict."""
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load metadata: {e}")
            return []
    return []


def save_metadata(metadata_list):
    """Save metadata to JSON file."""
    try:
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata_list, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Could not save metadata: {e}")
        return False


# Title and instructions
st.markdown(
    """
    <style>
    .main {
        max-width: 500px;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🕳️ Pothole Reporter")
st.markdown("Upload a photo of a pothole to help us maintain road infrastructure.")

# Upload section
st.markdown("### 📸 Upload Image")
uploaded_file = st.file_uploader(
    "Choose a pothole image",
    type=["jpg", "jpeg", "png", "bmp"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    # Display preview
    uploaded_file.seek(0)  # Reset file pointer to beginning
    image = Image.open(uploaded_file)
    st.image(image, caption="Preview", use_container_width=True)
    uploaded_file.seek(0)  # Reset again for later reading
    
    # Generate unique ID from timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    filename = f"pothole_{timestamp}.jpg"
    
    # Ask user for location using browser geolocation
    st.markdown("### 📍 Report Location")
    st.info("Click 'Get Location' to use your device's GPS, or enter coordinates manually")
    
    # Get geolocation from browser
    location = streamlit_geolocation()
    
    # Set default coordinates
    latitude = 48.1351
    longitude = 11.5820
    
    # Check if location was successfully retrieved
    if location:
        try:
            if "latitude" in location and "longitude" in location:
                latitude = location["latitude"]
                longitude = location["longitude"]
                accuracy = location.get("accuracy", "unknown")
                st.success(f"✅ Location captured (accuracy: ±{accuracy:.0f}m)")
            else:
                st.warning("Browser geolocation not available. Please enter coordinates manually or enable location access.")
        except (KeyError, TypeError):
            st.warning("Could not parse location. Please enter coordinates manually or enable location access.")
    else:
        st.warning("Browser geolocation not available. Please enter coordinates manually or enable location access.")
    
    # Allow manual override
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input(
            "Latitude",
            value=latitude,
            format="%.6f",
            label_visibility="collapsed",
        )
    with col2:
        longitude = st.number_input(
            "Longitude",
            value=longitude,
            format="%.6f",
            label_visibility="collapsed",
        )
    
    # Create metadata entry with user-provided coordinates
    metadata_entry = {
        "image": filename,
        "image_id": timestamp,
        "captured_at": datetime.now().isoformat(),
        "geometry": {
            "type": "Point",
            "coordinates": [longitude, latitude]
        },
        "labels": [
            {
                "label": 3,  # Pothole
                "confidence": 0.95,
                "avg_confidence": 0.95,
                "count": 1
            }
        ]
    }
    
    # Upload button
    if st.button("✅ Upload & Save", use_container_width=True):
        try:
            # Save image directly from uploaded file
            image_path = IMAGES_PRE_USERS_PATH / filename
            uploaded_file.seek(0)  # Reset file pointer
            with open(image_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Update metadata
            metadata_list = load_metadata()
            metadata_list.append(metadata_entry)
            
            if save_metadata(metadata_list):
                # Send signal to main app for immediate refresh
                SIGNAL_FILE = Path(__file__).parent / ".upload_signal"
                try:
                    SIGNAL_FILE.touch()  # Create signal file
                except:
                    pass  # Silently fail if can't create signal
                
                st.success(f"✅ Upload successful!")
                st.markdown(f"""
                **Image saved:** `{filename}`  
                **ID:** `{timestamp}`  
                **Location:** {metadata_entry['geometry']['coordinates']}
                """)
                
                st.info("🔄 The main dashboard will automatically detect and process this image. Go back to refresh the map!")
                
                # Clear file uploader
                st.rerun()
            else:
                st.error("Failed to save metadata")
        except Exception as e:
            st.error(f"Upload failed: {e}")
    
    # Clean up temp file
    try:
        os.remove(temp_path)
    except:
        pass

st.markdown("---")
st.markdown("### 📊 Upload Statistics")

metadata_list = load_metadata()
if metadata_list:
    st.metric("Total Uploads", len(metadata_list))
    
    # Count by date
    from collections import defaultdict
    uploads_by_date = defaultdict(int)
    for entry in metadata_list:
        captured_at = entry.get("captured_at", "")
        if isinstance(captured_at, str) and captured_at:
            date = captured_at.split("T")[0]
            if date:
                uploads_by_date[date] += 1
    
    if uploads_by_date:
        st.markdown("**Recent uploads:**")
        for date in sorted(uploads_by_date.keys(), reverse=True)[:7]:
            st.text(f"  {date}: {uploads_by_date[date]} potholes reported")
else:
    st.info("No uploads yet. Be the first to report a pothole!")

st.markdown("---")
st.caption("🗺️ Your reports help improve road maintenance. Thank you!")
