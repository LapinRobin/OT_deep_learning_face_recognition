import cv2
import numpy as np
import os
from tqdm import tqdm

def get_region_proposals(image, min_size=20, max_proposals=500):
    """
    Generate region proposals using Selective Search.
    Uses quality mode for better proposals.
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

def draw_region_proposals(image, regions, max_regions=100):
    """
    Draw region proposals on the image with different colors.
    Shows more regions (100) for better visualization.
    """
    vis_image = image.copy()
    np.random.seed(42)  # For consistent colors
    
    # Generate a fixed set of distinct colors
    colors = []
    for _ in range(max_regions):
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        colors.append(color)
    
    # Draw all regions with different colors and thickness based on size
    for i, (x, y, w, h) in enumerate(regions[:max_regions]):
        # Calculate thickness based on region size (larger regions get thicker lines)
        area = w * h
        thickness = max(1, min(3, int(np.log10(area) - 3)))
        
        # Draw rectangle with random color
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), colors[i % len(colors)], thickness)
        
        # Optionally add region number for reference
        if w > 50 and h > 50:  # Only add numbers to larger regions
            cv2.putText(vis_image, f'{i}', (x+5, y+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i % len(colors)], 1)
    
    # Add text showing total number of proposals
    total_proposals = len(regions)
    shown_proposals = min(total_proposals, max_regions)
    cv2.putText(vis_image, f'Showing {shown_proposals} of {total_proposals} proposals', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return vis_image

def process_image(image_path, output_dir):
    """Process a single image and visualize its region proposals."""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Get region proposals
    regions = get_region_proposals(image)
    
    # Create visualization
    vis_image = draw_region_proposals(image, regions)
    
    # Save visualization
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f'regions_{base_name}')
    cv2.imwrite(output_path, vis_image)
    
    return len(regions)

def main():
    # Directory setup
    input_dir = 'test_images_large'
    output_dir = 'region_proposals_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all images
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.pgm'))]
    
    for image_file in tqdm(image_files, desc='Processing images'):
        image_path = os.path.join(input_dir, image_file)
        num_proposals = process_image(image_path, output_dir)
        print(f'Processed {image_file}: Generated {num_proposals} region proposals')

if __name__ == '__main__':
    main() 