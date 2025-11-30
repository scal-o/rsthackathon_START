# Quick Start Guide - Inference Control with Streamlit

## Setup (One-time)

### 1. Activate Environment
```bash
conda activate rsthackathon
```

### 2. Configure Mapillary API (Optional)
Create/edit `.streamlit/secrets.toml`:
```toml
MAPILLARY_API_KEY = "your_api_key_here"
```

Get free API key: https://www.mapillary.com/developer

## Running the App

```bash
streamlit run streamlit_app.py
```

The app will:
- Open automatically in your browser (http://localhost:8501)
- Start centered on **Paris**
- Load pre-existing markers from `data/points.geojson`

---

## Mode 1: Pre-loaded Dataset (Default)

**What it does:**
- Shows markers from your local GeoJSON file
- Runs inference on images in `static/images/pre/`

**Steps:**
1. App starts with Paris view and pre-loaded markers
2. Use sidebar to **filter by detection type** (optional)
3. Click **"Run Inference on Pre-loaded Images"** button
4. Wait for processing
5. Click markers to see details and detected images

---

## Mode 2: Download & Process New Images

**What it does:**
- Downloads street view images from Mapillary
- Replaces images in `static/images/pre/`
- Runs inference automatically

**Steps:**

1. **Switch Mode:**
   - Sidebar → Choose "Mapillary API"

2. **Search Location:**
   - Enter city name (e.g., "Paris", "Tokyo")
   - Click "Search" button

3. **Set Download Area:**
   - Use sidebar slider to set radius (0.1 - 5.0 km)
   - Default: 1.0 km

4. **Place Pin:**
   - Click anywhere on the map to mark location

5. **Download & Process:**
   - Click **"⬇️ Download & Process"** button
   - Wait for download (status shown in app)
   - Inference runs automatically
   - Results appear on map

6. **Adjust Confidence:**
   - Use sidebar slider to filter detections (0.0 - 1.0)

---

## Key Settings (Sidebar)

### For Pre-loaded Mode:
- **Confidence Threshold**: Adjust what counts as a detection
- **Filter by Detection Type**: Show/hide specific damage types:
  - 🔴 Crack (small surface damage)
  - 🔵 Manhole (utility covers)
  - 🟠 Pothole (pavement holes)

### For Mapillary Mode:
- **Download Radius**: How far from clicked point to search (0.1 - 5.0 km)
- **Confidence Threshold**: Same as above

---

## Understanding Results

### On the Map:
- **Red markers** = Detection locations
- Click marker to see:
  - Image coordinates
  - Detection image with boxes
  - Confidence scores
  - Detection count per type

### Summary Panel:
- **Total Images**: Number of images processed
- **Total Detections**: Number of road issues found
- **Avg Confidence**: Average detection confidence
- **Labels Found**: Types of damage detected

---

## Example Workflows

### Quick Test (Pre-loaded)
```
1. Launch app
2. Click "Run Inference on Pre-loaded Images"
3. View results (2-5 min depending on image count)
4. Explore markers
```

### Survey New Area
```
1. Launch app
2. Switch to "Mapillary API" mode
3. Search for "Tokyo"
4. Click in Shibuya district
5. Set radius to 0.5 km
6. Click "Download & Process"
7. Wait for download & analysis
8. Review findings
```

### Compare Multiple Locations
```
1. Pre-loaded mode: Run inference on current dataset
2. Note findings
3. Switch to Mapillary mode
4. Download data for new location
5. Compare results
```

---

## Troubleshooting

### App won't start
```bash
# Check environment
conda activate rsthackathon

# Check dependencies
pip list | grep -E "streamlit|folium|requests"

# Verify conda environment
conda info --envs
```

### No images found
- **Pre-loaded mode**: Ensure images exist in `static/images/pre/`
- **Mapillary mode**: 
  - Check API key in `.streamlit/secrets.toml`
  - Verify location has Mapillary coverage
  - Try smaller radius or different location

### Inference fails
- Check model file exists: `models/modello_del_peter.onnx`
- Ensure images are JPEG, PNG, or BMP format
- Try with different confidence threshold

### Slow performance
- Close other applications
- Use smaller download radius (Mapillary mode)
- Check internet connection
- GPU recommended for faster inference

---

## Files Modified

- `streamlit_app.py` - Main interface with inference controls
- `inference.py` - No changes (but now imported by streamlit)
- `download_mapillary.py` - Called automatically by streamlit
- `STREAMLIT_WORKFLOW.md` - Detailed documentation

---

## For Advanced Users

### Manual Commands

```bash
# Run inference standalone
python inference.py

# Download images manually
python download_mapillary.py 48.8566 2.3522 1.0

# Check environment
python -c "from inference import process_image_batch; print('OK')"
```

### File Locations
- Pre-images: `static/images/pre/`
- Pre-metadata: `static/images/pre/metadata.json`
- Output images: `static/images/output/`
- Detection results: `static/images/output/detections.json`
- Model: `models/modello_del_peter.onnx`
- Pre-loaded markers: `data/points.geojson`

---

## Support

For detailed documentation, see: `STREAMLIT_WORKFLOW.md`

For issues:
1. Check console output for error messages
2. Review troubleshooting section above
3. Verify all dependencies are installed
4. Check file permissions and paths
