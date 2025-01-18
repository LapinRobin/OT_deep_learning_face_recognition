import cv2
import numpy as np
import os
from tqdm import tqdm
import math

def load_face_cascade():
    """Load OpenCV's pre-trained Haar Cascade face detector."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise ValueError("Error loading face cascade classifier")
    return face_cascade

def get_region_proposals(image, min_size=20, max_proposals=500):
    """
    Generate region proposals using Selective Search.
    We use 'selectiveSearchQuality' for better proposals.
    """
    # Create Selective Search Segmentation object
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    
    # Use quality mode for better proposals
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    
    # Filter proposals based on size and limit the number
    filtered_rects = []
    for (x, y, w, h) in rects:
        if w >= min_size and h >= min_size:
            filtered_rects.append((x, y, w, h))
            if len(filtered_rects) >= max_proposals:
                break
    
    return filtered_rects

def draw_region_proposals(image, regions, max_regions=50):
    """Draw region proposals on the image with different colors."""
    vis_image = image.copy()
    np.random.seed(42)  # For consistent colors
    
    # Only visualize a subset of regions to avoid cluttering
    for i, (x, y, w, h) in enumerate(regions[:max_regions]):
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 1)
    
    return vis_image

def process_regions(image_path, face_cascade, min_neighbors=5, visualize_proposals=True):
    """
    Process an image using region proposals and Haar Cascade face detection.
    Returns both detection results and visualization of region proposals.
    """
    # Read image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to grayscale for face detection
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Get region proposals
    regions = get_region_proposals(original_image)
    
    # Create visualization of region proposals
    if visualize_proposals:
        proposal_vis = draw_region_proposals(original_image, regions)
    else:
        proposal_vis = None
    
    detections = []
    
    for (x, y, w, h) in tqdm(regions, desc=f'Processing {os.path.basename(image_path)}'):
        # Extract region from grayscale image
        region = gray_image[y:y+h, x:x+w]
        
        # Skip if region too small
        if region.shape[0] < 30 or region.shape[1] < 30:
            continue
        
        faces = face_cascade.detectMultiScale(
            region,
            scaleFactor=1.1,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        
        for (rx, ry, rw, rh) in faces:
            abs_x = x + rx
            abs_y = y + ry
            
            detections.append({
                'x': abs_x,
                'y': abs_y,
                'width': rw,
                'height': rh,
                'confidence': 1.0
            })
    
    return original_image, detections, proposal_vis

def create_grid_visualization(images_data, grid_size=None):
    """
    Create a grid visualization of multiple images.
    images_data: list of tuples (original_image, proposal_vis, result_image)
    """
    if not grid_size:
        # Calculate grid size based on number of images
        n_images = len(images_data)
        grid_width = min(4, n_images)
        grid_height = math.ceil(n_images / grid_width)
    else:
        grid_width, grid_height = grid_size
    
    # Find maximum dimensions
    max_h = max(img[0].shape[0] for img in images_data)
    max_w = max(img[0].shape[1] for img in images_data)
    
    # Scale factor to make the grid fit in a reasonable size
    scale_factor = min(1.0, 1920 / (grid_width * max_w))
    
    # Create the grid
    grid_h = int(grid_height * max_h * scale_factor)
    grid_w = int(grid_width * max_w * scale_factor)
    grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for idx, (original, proposals, result) in enumerate(images_data):
        # Calculate position in grid
        grid_x = idx % grid_width
        grid_y = idx // grid_width
        
        # Resize images
        h, w = original.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Resize all three images
        original_resized = cv2.resize(original, (new_w, new_h))
        proposals_resized = cv2.resize(proposals, (new_w, new_h))
        result_resized = cv2.resize(result, (new_w, new_h))
        
        # Combine the three images horizontally
        combined = np.hstack([original_resized, proposals_resized, result_resized])
        
        # Calculate position in grid
        y_start = grid_y * new_h
        x_start = grid_x * (new_w * 3)  # Multiply by 3 because we have 3 images side by side
        
        # Place in grid
        if y_start + new_h <= grid_h and x_start + (new_w * 3) <= grid_w:
            grid_image[y_start:y_start + new_h, x_start:x_start + (new_w * 3)] = combined
    
    return grid_image

def non_max_suppression(detections, overlap_thresh=0.3):
    """Apply non-maximum suppression to reduce overlapping boxes."""
    if not detections:
        return []
    
    boxes = np.array([[d['x'], d['y'], d['x'] + d['width'], d['y'] + d['height']] 
                      for d in detections])
    
    # For Haar cascade, we don't have confidence scores, so we'll use the area as a proxy
    # Larger faces are typically more reliable detections
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idxs = np.argsort(areas)[::-1]
    
    pick = []
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
        
        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        # Compute overlap ratio
        overlap = (w * h) / areas[idxs[1:]]
        
        # Remove indices that exceed overlap threshold
        idxs = np.delete(
            idxs, 
            np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1))
        )
    
    return [detections[i] for i in pick]

def draw_detections(image, detections):
    """Draw bounding boxes for detected faces."""
    for det in detections:
        x, y = det['x'], det['y']
        w, h = det['width'], det['height']
        
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add label (optional since we don't have confidence scores from Haar cascade)
        cv2.putText(image, 'Face', (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return image

def main():
    # Load Haar Cascade classifier
    face_cascade = load_face_cascade()
    print("Loaded Haar Cascade face detector")
    
    # Directory setup
    input_dir = 'test_images_large'
    output_dir = 'region_detection_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all images
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.pgm'))]
    all_results = []
    
    for image_file in tqdm(image_files, desc='Processing images'):
        image_path = os.path.join(input_dir, image_file)
        
        # Process image with region proposals and Haar cascade
        original_image, detections, proposal_vis = process_regions(image_path, face_cascade)
        
        # Apply non-maximum suppression
        filtered_detections = non_max_suppression(detections, overlap_thresh=0.3)
        
        # Draw detections
        result_image = draw_detections(original_image.copy(), filtered_detections)
        
        # Save individual result
        output_path = os.path.join(output_dir, f'detected_{image_file}')
        cv2.imwrite(output_path, result_image)
        
        # Store results for grid visualization
        all_results.append((original_image, proposal_vis, result_image))
        
        print(f'Processed {image_file}: Found {len(filtered_detections)} faces')
    
    # Create and save grid visualization
    # grid_image = create_grid_visualization(all_results)
    # grid_output_path = os.path.join(output_dir, 'combined_results_grid.jpg')
    # cv2.imwrite(grid_output_path, grid_image)
    # print(f'Saved combined visualization to {grid_output_path}')

if __name__ == '__main__':
    main()