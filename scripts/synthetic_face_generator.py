"""
Synthetic Face Generator for Robustness Benchmarking

Generates synthetic face-like images under various challenging conditions:
- Lighting variations (dark, normal, bright, uneven)
- Face pose/angle variations (±45°, ±30°, ±15°, frontal)
- Occlusions (glasses, mask, hand)

Used for reproducible robustness testing without requiring actual face datasets.
All images are deterministic (seeded) for reproducibility.
"""

import os
import numpy as np
import cv2
from pathlib import Path


class SyntheticFaceGenerator:
    """Generate synthetic face-like patterns under various conditions."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        self.rng = np.random.RandomState(seed)
        self.seed = seed
    
    def generate_face_template(self, height: int = 224, width: int = 224) -> np.ndarray:
        """Generate a base face-like template using simple patterns.
        
        Creates a synthetic face using:
        - Gaussian blobs for eyes
        - Curve for nose
        - Rectangle for mouth
        - Ellipse for face outline
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            numpy array (H, W, 3) uint8 BGR image
        """
        img = np.ones((height, width, 3), dtype=np.uint8) * 200  # Base skin tone
        
        center_x = width // 2
        center_y = height // 2
        
        # Face outline (ellipse)
        cv2.ellipse(img, (center_x, center_y), (width//3, height//2.5), 0, 0, 360, (180, 140, 100), -1)
        
        # Left eye
        eye_left_x = center_x - width // 6
        eye_left_y = center_y - height // 5
        cv2.circle(img, (eye_left_x, eye_left_y), width // 20, (50, 50, 50), -1)
        cv2.circle(img, (eye_left_x - width // 40, eye_left_y - width // 40), width // 50, (255, 255, 255), -1)
        
        # Right eye
        eye_right_x = center_x + width // 6
        eye_right_y = center_y - height // 5
        cv2.circle(img, (eye_right_x, eye_right_y), width // 20, (50, 50, 50), -1)
        cv2.circle(img, (eye_right_x + width // 40, eye_right_y - width // 40), width // 50, (255, 255, 255), -1)
        
        # Nose (triangle)
        nose_points = np.array([
            [center_x, center_y - height // 10],
            [center_x - width // 30, center_y + height // 10],
            [center_x + width // 30, center_y + height // 10]
        ], dtype=np.int32)
        cv2.polylines(img, [nose_points], False, (150, 100, 80), 2)
        
        # Mouth
        mouth_y = center_y + height // 4
        cv2.ellipse(img, (center_x, mouth_y), (width // 8, height // 20), 0, 0, 180, (100, 50, 50), 2)
        
        # Add texture with noise
        noise = self.rng.normal(0, 8, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def apply_lighting(self, img: np.ndarray, condition: str) -> np.ndarray:
        """Apply lighting condition to image.
        
        Args:
            img: Input image
            condition: 'dark', 'normal', 'bright', 'uneven'
            
        Returns:
            Adjusted image
        """
        img_f = img.astype(np.float32)
        h, w = img.shape[:2]
        
        if condition == 'dark':
            # Dark lighting: multiply by 0.3-0.4
            factor = self.rng.uniform(0.3, 0.4)
            img_f = img_f * factor
            # Add some noise typical of low-light
            noise = self.rng.normal(0, 15, img.shape)
            img_f = np.clip(img_f + noise, 0, 255)
        
        elif condition == 'normal':
            # Normal lighting: no adjustment or slight adjustment
            factor = self.rng.uniform(0.95, 1.05)
            img_f = img_f * factor
        
        elif condition == 'bright':
            # Bright lighting: multiply by 1.4-1.6, clip at 255
            factor = self.rng.uniform(1.4, 1.6)
            img_f = np.clip(img_f * factor, 0, 255)
        
        elif condition == 'uneven':
            # Uneven lighting: bright spot on one side
            x = np.linspace(-1, 1, w)
            y = np.linspace(-1, 1, h)
            xv, yv = np.meshgrid(x, y)
            # Gaussian gradient (bright top-left, dark bottom-right)
            gradient = 1.0 + 0.5 * np.exp(-(xv**2 + yv**2) / 0.3)
            gradient = np.stack([gradient] * 3, axis=2)
            img_f = np.clip(img_f * gradient, 0, 255)
        
        return np.clip(img_f, 0, 255).astype(np.uint8)
    
    def apply_pose_rotation(self, img: np.ndarray, angle_deg: float) -> np.ndarray:
        """Apply head pose rotation via affine transform.
        
        Args:
            img: Input image
            angle_deg: Rotation angle in degrees (-45 to +45)
            
        Returns:
            Rotated image
        """
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        
        # Apply rotation (borders filled with white)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
        return rotated
    
    def apply_occlusion(self, img: np.ndarray, occlusion_type: str) -> np.ndarray:
        """Apply occlusion to image.
        
        Args:
            img: Input image
            occlusion_type: 'none', 'glasses', 'mask', 'hand', 'partial'
            
        Returns:
            Occluded image
        """
        if occlusion_type == 'none':
            return img.copy()
        
        h, w = img.shape[:2]
        center_x = w // 2
        center_y = h // 2
        result = img.copy()
        
        if occlusion_type == 'glasses':
            # Dark bands over eyes
            y1 = center_y - h // 5
            y2 = center_y - h // 7
            # Left eye region
            cv2.ellipse(result, (center_x - w // 6, y1), (w // 8, h // 15), 0, 0, 360, (0, 0, 0), -1)
            # Right eye region
            cv2.ellipse(result, (center_x + w // 6, y1), (w // 8, h // 15), 0, 0, 360, (0, 0, 0), -1)
        
        elif occlusion_type == 'mask':
            # Medical mask: cover lower half of face
            mask_top = center_y
            mask_bottom = center_y + h // 3
            cv2.rectangle(result, (center_x - w // 4, mask_top), (center_x + w // 4, mask_bottom), (100, 100, 100), -1)
        
        elif occlusion_type == 'hand':
            # Hand covering lower right
            x1 = center_x + w // 6
            y1 = center_y
            x2 = center_x + w // 2
            y2 = center_y + h // 3
            cv2.rectangle(result, (x1, y1), (x2, y2), (120, 80, 60), -1)  # Skin tone
        
        elif occlusion_type == 'partial':
            # Partial occlusion: random small rectangles
            num_patches = 2
            for _ in range(num_patches):
                x = self.rng.randint(w // 4, 3 * w // 4)
                y = self.rng.randint(h // 4, 3 * h // 4)
                size = self.rng.randint(h // 10, h // 5)
                cv2.rectangle(result, (x, y), (x + size, y + size), (0, 0, 0), -1)
        
        return result
    
    def apply_blur(self, img: np.ndarray, blur_level: str = 'normal') -> np.ndarray:
        """Apply blur to simulate motion or focus issues.
        
        Args:
            img: Input image
            blur_level: 'none', 'slight', 'moderate', 'severe'
            
        Returns:
            Blurred image
        """
        if blur_level == 'none':
            return img.copy()
        
        kernel_sizes = {
            'slight': 3,
            'moderate': 7,
            'severe': 15
        }
        
        kernel_size = kernel_sizes.get(blur_level, 3)
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    def generate(self,
                 lighting: str = 'normal',
                 pose_angle: float = 0.0,
                 occlusion: str = 'none',
                 blur: str = 'none',
                 height: int = 224,
                 width: int = 224) -> np.ndarray:
        """Generate a complete synthetic face image.
        
        Args:
            lighting: 'dark', 'normal', 'bright', 'uneven'
            pose_angle: Rotation angle in degrees
            occlusion: 'none', 'glasses', 'mask', 'hand', 'partial'
            blur: 'none', 'slight', 'moderate', 'severe'
            height: Image height
            width: Image width
            
        Returns:
            Synthetic face image (H, W, 3) uint8
        """
        # Start with base template
        img = self.generate_face_template(height, width)
        
        # Apply transformations
        img = self.apply_lighting(img, lighting)
        if pose_angle != 0:
            img = self.apply_pose_rotation(img, pose_angle)
        img = self.apply_occlusion(img, occlusion)
        img = self.apply_blur(img, blur)
        
        return img
    
    def generate_dataset(self,
                        output_dir: str,
                        num_per_condition: int = 5,
                        height: int = 224,
                        width: int = 224) -> dict:
        """Generate a complete robustness benchmark dataset.
        
        Args:
            output_dir: Directory to save images
            num_per_condition: Number of images per condition
            height: Image height
            width: Image width
            
        Returns:
            Dictionary mapping condition names to image paths and metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define test conditions
        lighting_conditions = ['dark', 'normal', 'bright', 'uneven']
        pose_angles = [-45, -30, -15, 0, 15, 30, 45]
        occlusions = ['none', 'glasses', 'mask', 'hand', 'partial']
        
        metadata = {
            'seed': self.seed,
            'height': height,
            'width': width,
            'conditions': {}
        }
        
        image_count = 0
        
        # Lighting variations
        for lighting in lighting_conditions:
            condition_key = f'lighting_{lighting}'
            metadata['conditions'][condition_key] = []
            for i in range(num_per_condition):
                img = self.generate(lighting=lighting, height=height, width=width)
                filename = f"{condition_key}_{i:03d}.png"
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), img)
                metadata['conditions'][condition_key].append(str(filename))
                image_count += 1
        
        # Pose variations
        for angle in pose_angles:
            condition_key = f'pose_{angle:+d}deg'
            metadata['conditions'][condition_key] = []
            for i in range(num_per_condition):
                img = self.generate(pose_angle=angle, height=height, width=width)
                filename = f"pose_{angle:+d}deg_{i:03d}.png"
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), img)
                metadata['conditions'][condition_key].append(str(filename))
                image_count += 1
        
        # Occlusion variations
        for occlusion in occlusions:
            condition_key = f'occlusion_{occlusion}'
            metadata['conditions'][condition_key] = []
            for i in range(num_per_condition):
                img = self.generate(occlusion=occlusion, height=height, width=width)
                filename = f"occlusion_{occlusion}_{i:03d}.png"
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), img)
                metadata['conditions'][condition_key].append(str(filename))
                image_count += 1
        
        # Combined challenging conditions
        challenging_cases = [
            {'lighting': 'dark', 'occlusion': 'glasses', 'name': 'dark_glasses'},
            {'lighting': 'bright', 'pose_angle': 30, 'name': 'bright_angle'},
            {'lighting': 'uneven', 'occlusion': 'mask', 'name': 'uneven_mask'},
            {'pose_angle': -45, 'occlusion': 'partial', 'name': 'extreme_angle_partial'},
        ]
        
        condition_key = 'combined_challenging'
        metadata['conditions'][condition_key] = []
        for case in challenging_cases:
            for i in range(num_per_condition):
                name = case.pop('name')
                img = self.generate(**case, height=height, width=width)
                filename = f"combined_{name}_{i:03d}.png"
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), img)
                metadata['conditions'][condition_key].append(str(filename))
                image_count += 1
        
        metadata['total_images'] = image_count
        return metadata


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic face dataset for robustness testing')
    parser.add_argument('--output', default='data/synthetic_faces', help='Output directory')
    parser.add_argument('--num-per-condition', type=int, default=5, help='Number of images per condition')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--height', type=int, default=224, help='Image height')
    parser.add_argument('--width', type=int, default=224, help='Image width')
    
    args = parser.parse_args()
    
    generator = SyntheticFaceGenerator(seed=args.seed)
    metadata = generator.generate_dataset(
        output_dir=args.output,
        num_per_condition=args.num_per_condition,
        height=args.height,
        width=args.width
    )
    
    print(f"✓ Generated {metadata['total_images']} synthetic images in {args.output}")
    print(f"  Conditions: {len(metadata['conditions'])}")
    for condition_name, images in metadata['conditions'].items():
        print(f"    - {condition_name}: {len(images)} images")
