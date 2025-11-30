# Streamlit Workflow - Inference Control Guide

## Overview
The updated Streamlit application provides an integrated workflow for road damage detection using either pre-loaded images or Mapillary street view data.

## Two Operation Modes

### 1. **Local GeoJSON Mode** (Default)
This mode works with your pre-loaded dataset:

**Workflow:**
1. App starts centered on Paris
2. Pre-loaded markers are displayed from `data/points.geojson`
3. Each marker represents a location with detected road damage (cracks, manholes, potholes)
4. Click **"Run Inference on Pre-loaded Images"** button to reprocess all images in `static/images/pre/`
5. Inference results are displayed on the map
6. Click markers to see:
   - Image coordinates
   - Detection details (type, confidence)
   - Detected image with bounding boxes

**Files Used:**
- Input images: `static/images/pre/`
- Metadata: `static/images/pre/metadata.json`
- Output images: `static/images/output/`
- Detections: `static/images/output/detections.json`

**Filtering:**
- Use the sidebar to filter markers by detection type:
  - 🔴 Crack
  - 🔵 Manhole
  - 🟠 Pothole

---

### 2. **Mapillary API Mode**
This mode downloads fresh street view images from Mapillary:

**Workflow:**
1. Enter a city name in the search box (e.g., "Paris", "Tokyo")
2. Switch to "Mapillary API" mode in the sidebar
3. Click anywhere on the map to place a pin at your desired location
4. Configure the download radius (0.1 - 5.0 km, default 1 km)
5. Click **"Download & Process"** button
6. The app will:
   - Clear old images from `static/images/pre/`
   - Download new images from Mapillary (equivalent to: `python download_mapillary.py {lat} {lon} {radius}`)
   - Save metadata.json
   - Automatically run inference on downloaded images
7. View results on the map with the same detection details

**Settings:**
- Download radius: Controls how many images are fetched around the clicked location
- Confidence threshold: Adjustable in sidebar (0.0 - 1.0, default 0.74)
- Requires: Valid Mapillary API key in `.streamlit/secrets.toml`

---

## Configuration

### Mapillary API Key
Add to `.streamlit/secrets.toml`:
```toml
MAPILLARY_API_KEY = "your_api_key_here"
```

Get a free API key at: https://www.mapillary.com/developer

### Confidence Threshold
- Adjustable in the sidebar for both modes
- Range: 0.0 to 1.0
- Default: 0.74
- Only detections above this threshold are displayed

---

## Key Features

### 1. **Integrated Inference**
- Runs the ONNX model on images automatically
- Uses multi-image batch processing for efficiency
- Shows progress during processing

### 2. **Dynamic Data Loading**
- Switch between pre-loaded and Mapillary modes without restarting
- Automatically overwrites old images when downloading new ones
- Preserves detection history in JSON format

### 3. **Detection Summary**
After inference, displays:
- Total images processed
- Total detections found
- Average confidence score
- Unique detection labels

### 4. **Visual Feedback**
- Color-coded markers for different detection types:
  - Red (🔴): Cracks
  - Blue (🔵): Manholes
  - Orange (🟠): Potholes
- Marker popups show detected image with bounding boxes
- Detection statistics per location

---

## Model & Performance

**Model:** `models/modello_del_peter.onnx`
- ONNX format for cross-platform compatibility
- Optimized inference runtime

**Processing:**
- Batch processing for multiple images
- GPU acceleration (if available)
- Output includes:
  - Detected bounding boxes
  - Class labels (1: Crack, 2: Manhole, 3: Pothole)
  - Confidence scores

---

## Troubleshooting

### "No images found in pre directory"
- Ensure `static/images/pre/` contains image files
- Or use Mapillary mode to download new images

### Mapillary API errors
- Check if API key is valid in `.streamlit/secrets.toml`
- Verify internet connection
- Try with smaller radius (0.1-0.5 km)

### Inference not running
- Check if model file exists: `models/modello_del_peter.onnx`
- Verify image format (JPEG, PNG, BMP, TIFF)
- Check confidence threshold is reasonable

### Images not being downloaded
- Verify Mapillary API key is configured
- Check if location has available Mapillary coverage
- Increase download radius

---

## Command Reference

### Manual Commands

**Download images manually:**
```bash
python download_mapillary.py 48.8566 2.3522 1.0 --output static/images/pre
```

**Run inference manually:**
```bash
python inference.py
```

**Run Streamlit app:**
```bash
streamlit run streamlit_app.py
```

---

## File Structure
```
rsthackathon_START/
├── streamlit_app.py           # Main Streamlit interface
├── inference.py               # Inference logic
├── download_mapillary.py      # Mapillary downloader
├── models/
│   └── modello_del_peter.onnx # ONNX model
├── static/images/
│   ├── pre/                   # Input images
│   │   └── metadata.json      # Image metadata
│   └── output/                # Detected images
│       └── detections.json    # Detection results
├── data/
│   └── points.geojson         # Pre-loaded markers
└── .streamlit/
    └── secrets.toml           # API keys
```

---

## Example Use Cases

### Case 1: Quick Analysis of Pre-loaded Data
1. Launch app: `streamlit run streamlit_app.py`
2. Click "Run Inference on Pre-loaded Images"
3. Wait for results
4. Explore markers on map

### Case 2: New Location Survey
1. Launch app
2. Switch to "Mapillary API" mode
3. Search for city
4. Click on map to mark location
5. Adjust radius (default 1 km)
6. Click "Download & Process"
7. Wait for download and inference
8. Review results

### Case 3: Batch Processing Multiple Locations
1. Use pre-loaded mode
2. Run inference
3. Download different areas via Mapillary
4. Review each set of results

---

## Performance Notes

- Initial load: ~5-10 seconds
- Inference per image: ~0.5-2 seconds (depends on image size and model)
- Mapillary download: Depends on internet speed and number of images
- Peak memory usage: ~2-3 GB (with multiple concurrent processes)

Recommended: Run on machine with GPU for faster inference
