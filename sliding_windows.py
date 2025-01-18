import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from net import Net2
import os
from tqdm import tqdm
import cv2

def load_model(model_path='best_model.pth', device='cpu'):
    """Load the trained model."""
    model = Net2().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def sliding_window(image, window_size, step_size):
    """Yield sliding windows of the image."""
    for y in range(0, image.shape[0] - window_size[0] + 1, step_size[0]):
        for x in range(0, image.shape[1] - window_size[1] + 1, step_size[1]):
            yield (x, y, image[y:y + window_size[0], x:x + window_size[1]])

def process_image(image_path, model, device, conf_threshold=0.95):
    """Process a single image with multiple window sizes."""
    # Read and convert image to grayscale
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Define window sizes (you can adjust these)
    window_sizes = [(36, 36), (72, 72)]
    detections = []

    # Normalize transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,), std=(1,))
    ])

    # Process each window size
    for window_size in tqdm(window_sizes, desc='Processing window sizes'):
        # Adjust step size based on window size (e.g., 25% of window size)
        step_size = (max(1, window_size[0] // 4), max(1, window_size[1] // 4))
        
        # Calculate total number of windows for this size
        n_windows = ((gray_image.shape[0] - window_size[0]) // step_size[0] + 1) * \
                   ((gray_image.shape[1] - window_size[1]) // step_size[1] + 1)
        
        # Slide window over image
        for (x, y, window) in tqdm(sliding_window(gray_image, window_size, step_size), 
                                 total=n_windows,
                                 desc=f'Processing {window_size[0]}x{window_size[1]} windows',
                                 leave=False):
            # Resize window to model input size (36x36)
            resized = cv2.resize(window, (36, 36))
            
            # Convert to PIL Image and apply transforms
            img_tensor = transform(Image.fromarray(resized)).unsqueeze(0).to(device)
            
            # Get model prediction
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence = probabilities[0][1].item()  # Probability of face class
                
                if confidence > conf_threshold:
                    detections.append({
                        'x': x,
                        'y': y,
                        'width': window_size[0],
                        'height': window_size[1],
                        'confidence': confidence
                    })
    
    return original_image, detections

def non_max_suppression(detections, overlap_thresh=0.3):
    """Apply non-maximum suppression to avoid multiple detections of the same face."""
    if not detections:
        return []
    
    # Convert to numpy array for easier processing
    boxes = np.array([[d['x'], d['y'], d['x'] + d['width'], d['y'] + d['height']] for d in detections])
    scores = np.array([d['confidence'] for d in detections])
    
    # Compute area of each box
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Sort by confidence
    idxs = np.argsort(scores)[::-1]
    
    pick = []
    while len(idxs) > 0:
        # Pick the box with highest confidence
        last = len(idxs) - 1
        i = idxs[0]
        pick.append(i)
        
        # Find the intersection
        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
        
        # Compute intersection area
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / area[idxs[1:]]
        
        # Delete all indexes from the index list that have high overlap
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))
    
    return [detections[i] for i in pick]

def draw_detections(image, detections):
    """Draw bounding boxes for detected faces."""
    for det in detections:
        x, y = det['x'], det['y']
        w, h = det['width'], det['height']
        conf = det['confidence']
        
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Add confidence score
        cv2.putText(image, f'{conf:.2f}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    
    # Load model
    model = load_model(device=device)
    
    # Directory containing images to process
    input_dir = 'test_images_large'  # Change this to your input directory
    output_dir = 'detection_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all images
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.pgm'))]
    
    for image_file in tqdm(image_files, desc='Processing images'):
        image_path = os.path.join(input_dir, image_file)
        
        # Process image and get detections
        original_image, detections = process_image(image_path, model, device)
        
        # Apply non-maximum suppression
        filtered_detections = non_max_suppression(detections)
        
        # Draw detections
        result_image = draw_detections(original_image.copy(), filtered_detections)
        
        # Save result
        output_path = os.path.join(output_dir, f'detected_{image_file}')
        cv2.imwrite(output_path, result_image)
        
        print(f'Processed {image_file}: Found {len(filtered_detections)} faces')

if __name__ == '__main__':
    main() 