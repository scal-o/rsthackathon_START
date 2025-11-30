import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont


def preprocess_image(image_path, input_shape):
    """
    Preprocess image for ONNX model input.

    Args:
        image_path: Path to input image
        input_shape: Expected input shape (batch, channels, height, width)

    Returns:
        Preprocessed image array, resized image (for drawing), and original size
    """
    # Load image
    img = Image.open(image_path).convert("RGB")  # Convert to RGB to handle RGBA/grayscale
    original_size = img.size  # (width, height)

    # Determine if model expects CHW or HWC format based on input shape
    # Common formats:
    # - CHW: [batch, channels, height, width] e.g., [1, 3, 224, 224]
    # - HWC: [batch, height, width, channels] e.g., [1, 224, 224, 3]

    if input_shape[1] == 3 or input_shape[1] == 1:
        # CHW format: [batch, channels, height, width]
        target_height, target_width = input_shape[2], input_shape[3]
        use_chw = True
    else:
        # HWC format: [batch, height, width, channels]
        target_height, target_width = input_shape[1], input_shape[2]
        use_chw = False

    # Resize image
    resized_img = img.resize((target_width, target_height))
    img = resized_img

    # Convert to numpy array
    img_array = np.array(img).astype(np.float32)

    # Normalize to [0, 1]
    img_array = img_array / 255.0

    # Convert HWC to CHW format if needed
    if use_chw:
        img_array = img_array.transpose(2, 0, 1)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, resized_img


def draw_bounding_boxes(image, boxes, labels, scores, threshold=0.5):
    """
    Draw bounding boxes on image.

    Args:
        image: PIL Image
        boxes: Array of bounding boxes [x1, y1, x2, y2] or [x, y, w, h]
        labels: Array of class labels
        scores: Array of confidence scores
        threshold: Minimum confidence threshold

    Returns:
        Image with drawn bounding boxes
    """
    draw = ImageDraw.Draw(image)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Label color mapping
    LABEL_COLORS = {
        1: "#e74c3c",  # Crack - Red
        2: "#3498db",  # Manhole - Blue
        3: "#e67e22",  # Pothole - Orange
    }

    LABEL_NAMES = {1: "Crack", 2: "Manhole", 3: "Pothole"}

    img_width, img_height = image.size

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if score < threshold:
            continue

        # Parse box coordinates
        # Format: [x_center, y_center, width, height]
        if len(box) == 4:
            x_center, y_center, width, height = box

            # Convert from center format to corner format
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            # Clamp to image boundaries
            x1 = max(0, min(x1, img_width))
            x2 = max(0, min(x2, img_width))
            y1 = max(0, min(y1, img_height))
            y2 = max(0, min(y2, img_height))

            # Skip invalid boxes
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue

        # Get color based on label
        label_int = int(label)
        color = LABEL_COLORS.get(label_int, "#999999")  # Default gray if unknown
        label_name = LABEL_NAMES.get(label_int, f"Class {label_int}")

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label and score
        text = f"{label_name}: {score:.2f}"

        # Draw text background
        text_bbox = draw.textbbox((x1, y1 - 20), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 20), text, fill="white", font=font)

    return image


def load_metadata(metadata_path):
    """
    Load metadata.json and create a mapping of image IDs to coordinates.
    Handles both pre-loaded format (with 'id') and user-uploaded format (with 'image_id').

    Args:
        metadata_path: Path to metadata.json file

    Returns:
        Dictionary mapping image IDs to their metadata
    """
    if not Path(metadata_path).exists():
        print(f"Warning: Metadata file not found at {metadata_path}")
        return {}

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Create a mapping of image ID to metadata
    metadata_map = {}
    if isinstance(metadata, list):
        # If it's a list, map by 'id' field (pre-loaded) or 'image_id' field (user-uploaded)
        for entry in metadata:
            # Try both 'id' and 'image_id' keys
            key = entry.get("id") or entry.get("image_id")
            if key:
                metadata_map[key] = entry
    elif isinstance(metadata, dict):
        # If it's already a dict, check if it has an 'id' field or use keys directly
        if "id" in metadata:
            # Single entry dict
            metadata_map[metadata["id"]] = metadata
        else:
            # Already a mapping dict
            metadata_map = metadata

    return metadata_map


def extract_image_id(filename):
    """
    Extract image ID from filename.
    Handles two formats:
    - Pre-loaded: 3736697343123377_1578561568299.jpg -> extract first part "3736697343123377"
    - User-uploaded: pothole_20251130_091234_567.jpg -> extract timestamp part "20251130_091234_567"

    Args:
        filename: Image filename

    Returns:
        Image ID as string
    """
    stem = Path(filename).stem
    
    # Check if filename starts with "pothole_" (user-uploaded format)
    if stem.startswith("pothole_"):
        # For user uploads: pothole_YYYYMMDD_HHMMSS_mmm -> return YYYYMMDD_HHMMSS_mmm
        parts = stem.split("_")
        if len(parts) >= 4:
            # Skip "pothole" prefix and join the timestamp parts
            return "_".join(parts[1:])
    
    # For pre-loaded format: extract only the ID part before the first underscore
    return stem.split("_")[0]


def save_detections_to_json(
    image_path,
    boxes,
    labels,
    scores,
    metadata_map,
    output_json_path="detections.json",
    confidence_threshold=0.5,
):
    """
    Save detections to GeoJSON FeatureCollection format.

    Args:
        image_path: Path to the image file
        boxes: Array of bounding boxes
        labels: Array of class labels
        scores: Array of confidence scores
        metadata_map: Dictionary mapping image IDs to metadata
        output_json_path: Path to save GeoJSON file
        confidence_threshold: Minimum confidence threshold
    """
    # Filter detections by confidence threshold
    filtered_data = [
        (label, score) for label, score in zip(labels, scores) if score >= confidence_threshold
    ]

    if not filtered_data:
        return None

    # Extract image ID from filename
    image_id = extract_image_id(Path(image_path).name)

    # Get coordinates from metadata
    latitude = None
    longitude = None
    if image_id in metadata_map:
        metadata_entry = metadata_map[image_id]
        if "geometry" in metadata_entry and "coordinates" in metadata_entry["geometry"]:
            coords = metadata_entry["geometry"]["coordinates"]
            longitude = coords[0]  # GeoJSON format: [longitude, latitude]
            latitude = coords[1]
            print(f"  Found coordinates from metadata: ({latitude}, {longitude})")

    if latitude is None or longitude is None:
        print(f"  Warning: No metadata found for image {image_id}")
        return None

    # Aggregate labels
    label_confidences = {}
    for label, score in filtered_data:
        label_int = int(label)
        if label_int not in label_confidences:
            label_confidences[label_int] = []
        label_confidences[label_int].append(float(score))

    # Calculate statistics per label
    labels_info = [
        {
            "label": label,
            "avg_confidence": round(sum(scores) / len(scores), 4),
            "min_confidence": round(min(scores), 4),
            "max_confidence": round(max(scores), 4),
            "count": len(scores),
        }
        for label, scores in sorted(label_confidences.items())
    ]

    # Create GeoJSON feature
    feature = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
        "properties": {"image": Path(image_path).name, "image_id": image_id, "labels": labels_info},
    }

    # Load existing FeatureCollection if file exists
    if Path(output_json_path).exists():
        try:
            with open(output_json_path, "r") as f:
                geojson = json.load(f)
                if geojson.get("type") != "FeatureCollection":
                    geojson = {"type": "FeatureCollection", "features": []}
        except:
            geojson = {"type": "FeatureCollection", "features": []}
    else:
        geojson = {"type": "FeatureCollection", "features": []}

    # Add feature to collection
    geojson["features"].append(feature)

    # Update summary
    geojson["summary"] = {
        "total_images": len(geojson["features"]),
        "unique_labels": sorted(
            list(
                set(
                    lbl["label"]
                    for feat in geojson["features"]
                    for lbl in feat["properties"]["labels"]
                )
            )
        ),
        "total_detections": sum(
            lbl["count"] for feat in geojson["features"] for lbl in feat["properties"]["labels"]
        ),
        "confidence_stats": {
            "min": round(
                min(
                    lbl["min_confidence"]
                    for feat in geojson["features"]
                    for lbl in feat["properties"]["labels"]
                ),
                4,
            )
            if geojson["features"]
            else 0,
            "max": round(
                max(
                    lbl["max_confidence"]
                    for feat in geojson["features"]
                    for lbl in feat["properties"]["labels"]
                ),
                4,
            )
            if geojson["features"]
            else 0,
            "avg": round(
                sum(
                    lbl["avg_confidence"] * lbl["count"]
                    for feat in geojson["features"]
                    for lbl in feat["properties"]["labels"]
                )
                / sum(
                    lbl["count"]
                    for feat in geojson["features"]
                    for lbl in feat["properties"]["labels"]
                ),
                4,
            )
            if geojson["features"]
            else 0,
        },
    }

    # Save to GeoJSON
    with open(output_json_path, "w") as f:
        json.dump(geojson, f, indent=2)

    return feature


def process_image_batch(
    model_path,
    image_paths,
    metadata_map=None,
    output_dir=None,
    output_json_path=None,
    confidence_threshold=0.5,
    verbose=False,
):
    """
    Process multiple images with a single model session (efficient for batch processing).

    Args:
        model_path: Path to ONNX model
        image_paths: List of image paths to process
        metadata_map: Dictionary mapping image IDs to metadata (optional)
        output_dir: Directory to save output images (optional)
        output_json_path: Path to save GeoJSON output (optional)
        confidence_threshold: Minimum confidence threshold
        verbose: Print debug information

    Returns:
        List of dicts with detection results for each image
    """
    # Load model once for all images
    session = ort.InferenceSession(model_path)

    results = []

    for image_path in image_paths:
        try:
            # Determine output path
            output_image_path = None
            if output_dir:
                output_filename = f"{Path(image_path).stem}_detected{Path(image_path).suffix}"
                output_image_path = Path(output_dir) / output_filename

            # Run inference
            boxes, labels, scores, result_img = run_inference(
                model_path,
                str(image_path),
                output_image_path=str(output_image_path) if output_image_path else None,
                confidence_threshold=confidence_threshold,
                session=session,
                verbose=verbose,
            )

            # Save to JSON if requested
            feature = None
            if output_json_path and metadata_map:
                feature = save_detections_to_json(
                    str(image_path),
                    boxes,
                    labels,
                    scores,
                    metadata_map,
                    output_json_path=str(output_json_path),
                    confidence_threshold=confidence_threshold,
                )

            results.append(
                {
                    "image_path": str(image_path),
                    "boxes": boxes,
                    "labels": labels,
                    "scores": scores,
                    "result_img": result_img,
                    "feature": feature,
                }
            )

        except Exception as e:
            if verbose:
                print(f"Error processing {image_path}: {e}")
            results.append({"image_path": str(image_path), "error": str(e)})

    return results


def run_inference(
    model_path,
    image_path,
    output_image_path=None,
    confidence_threshold=0.5,
    session=None,
    verbose=True,
):
    """
    Run object detection inference and save result with bounding boxes.

    Args:
        model_path: Path to ONNX model
        image_path: Path to input image (can be str or PIL Image)
        output_image_path: Path to save output image (optional)
        confidence_threshold: Minimum confidence for detection
        session: Pre-loaded ONNX session (optional, for reuse)
        verbose: Print debug information

    Returns:
        tuple: (boxes, labels, scores, result_img)
    """
    # Load ONNX model if not provided
    if session is None:
        session = ort.InferenceSession(model_path)

    # Get model input details
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    if verbose:
        print(f"Model input: {input_name}")
        print(f"Input shape: {input_shape}")

        # Print all outputs
        print("\nModel outputs:")
        for i, output in enumerate(session.get_outputs()):
            print(f"  Output {i}: {output.name}, shape: {output.shape}")

    # Preprocess image
    img_array, original_img = preprocess_image(image_path, input_shape)

    # Run inference
    outputs = session.run(None, {input_name: img_array})

    # Parse outputs (format depends on your specific model)
    if verbose:
        print(f"\nNumber of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"Output {i} shape: {output.shape}")

    # Example parsing for common YOLO/SSD format
    if len(outputs) == 1:
        # Single output format [batch, num_detections, 6+]
        detections = outputs[0][0]  # Remove batch dimension
        if verbose:
            print(f"\nDetections shape after removing batch: {detections.shape}")
            print(f"First detection sample: {detections[0]}")

        boxes = detections[:, :4]  # First 4 values are box coordinates
        scores = detections[:, 4]  # 5th value is confidence
        labels = detections[:, 5]  # 6th value is class label
    else:
        # Multiple outputs format
        boxes = outputs[0][0]  # Assuming first output is boxes
        labels = outputs[1][0]  # Second output is labels
        scores = outputs[2][0]  # Third output is scores
        if verbose:
            print(f"\nBoxes shape: {boxes.shape}")
            print(f"First box sample: {boxes[0]}")

    if verbose:
        print(f"\nDetections found: {len(boxes)}")
        print(f"Score range: min={scores.min():.3f}, max={scores.max():.3f}")
        print(f"Box coordinate range: min={boxes.min():.3f}, max={boxes.max():.3f}")

    # Draw bounding boxes
    result_img = draw_bounding_boxes(
        original_img, boxes, labels, scores, threshold=confidence_threshold
    )

    # Save result if path provided
    if output_image_path:
        result_img.save(output_image_path)
        if verbose:
            print(f"\nResult saved to: {output_image_path}")

    # Return detections and result image
    return boxes, labels, scores, result_img


# Example usage
if __name__ == "__main__":
    from pathlib import Path

    model_path = "./models/modello_del_peter.onnx"
    data_dir = Path("static/images/pre")
    output_dir = Path("static/images/output")
    metadata_path = "./static/images/pre/metadata.json"  # Path to your metadata file
    confidence_threshold = 0.74

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Load metadata
    print("Loading metadata...")
    metadata_map = load_metadata(metadata_path)
    print(f"Loaded metadata for {len(metadata_map)} images\n")

    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    # Get all image files in data directory
    image_files = [
        f for f in data_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No image files found in {data_dir}")
        exit(1)

    print(f"Found {len(image_files)} images to process\n")

    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        print(f"{'=' * 60}")
        print(f"Processing {idx}/{len(image_files)}: {image_path.name}")
        print(f"{'=' * 60}")

        try:
            # Generate output filename
            output_filename = f"{image_path.stem}_detected{image_path.suffix}"
            output_path = output_dir / output_filename

            # Run inference
            boxes, labels, scores, result_img = run_inference(
                model_path,
                str(image_path),
                output_image_path=str(output_path),
                confidence_threshold=confidence_threshold,
            )

            # Print detections for this image
            detected = [
                (box, label, score)
                for box, label, score in zip(boxes, labels, scores)
                if score >= confidence_threshold
            ]

            if detected:
                print(f"\nDetected {len(detected)} objects:")
                for box, label, score in detected[:5]:  # Show first 5
                    print(f"  Label: {int(label)}, Score: {score:.3f}")
                if len(detected) > 5:
                    print(f"  ... and {len(detected) - 5} more")
            else:
                print("\nNo objects detected above threshold")

            # Save detections to JSON
            json_output_path = output_dir / "detections.json"
            detections_data = save_detections_to_json(
                str(image_path),
                boxes,
                labels,
                scores,
                metadata_map,
                output_json_path=str(json_output_path),
                confidence_threshold=confidence_threshold,
            )
            if detections_data:
                print(f"Detections saved to: {json_output_path}")
            else:
                print(f"No detections above threshold for this image")

            print()

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}\n")
            continue

    print(f"{'=' * 60}")
    print(f"Processing complete! Results saved to {output_dir}/")
    print(f"{'=' * 60}")

    # Print final summary
    json_output_path = output_dir / "detections.json"
    if json_output_path.exists():
        with open(json_output_path, "r") as f:
            final_data = json.load(f)
            if "summary" in final_data:
                print("\n" + "=" * 60)
                print("FINAL SUMMARY")
                print("=" * 60)
                print(f"Total images processed: {final_data['summary']['total_images']}")
                print(f"Unique labels detected: {final_data['summary']['unique_labels']}")
                print(f"Total detections: {final_data['summary']['total_detections']}")
                print(
                    f"Confidence range: {final_data['summary']['confidence_stats']['min']} - {final_data['summary']['confidence_stats']['max']}"
                )
                print(f"Average confidence: {final_data['summary']['confidence_stats']['avg']}")
                print("=" * 60)
