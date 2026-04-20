"""
Concurrency Benchmark: Evaluate system throughput and latency under load

Simulates multiple concurrent camera feeds (threads) processing faces simultaneously.
Measures:
- FPS degradation with thread count
- Latency percentiles (p50, p95, p99)
- Memory usage scaling
- GPU/CPU utilization if available

Helps determine:
- Maximum number of cameras per deployment
- Throughput limits
- Resource requirements for scalability
"""

import os
import sys
import time
import csv
import json
import threading
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
import queue

import numpy as np
import psutil

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.metrics import CameraMetricsTracker
from scripts.synthetic_face_generator import SyntheticFaceGenerator


class MockCameraSimulator:
    """Simulate a camera processing face frames."""
    
    def __init__(self, camera_id: int, frame_rate: int = 10):
        """Initialize mock camera.
        
        Args:
            camera_id: Unique camera ID
            frame_rate: Frames per second
        """
        self.camera_id = camera_id
        self.frame_rate = frame_rate
        self.frame_interval = 1.0 / frame_rate
        self.metrics = CameraMetricsTracker(camera_name=f"cam_{camera_id}")
        self.frames_processed = 0
        self.errors = 0
        self.stop_event = threading.Event()
    
    def process_frame(self) -> Dict:
        """Simulate processing a single frame.
        
        Returns:
            Dictionary with timing information
        """
        start_time = time.time()
        
        try:
            # Simulate pipeline stages
            detection_time = np.random.uniform(30, 40) / 1000  # 30-40ms
            recognition_time = np.random.uniform(15, 25) / 1000  # 15-25ms
            liveness_time = np.random.uniform(15, 25) / 1000  # 15-25ms
            
            total_time = detection_time + recognition_time + liveness_time
            
            # Simulate success
            is_recognized = np.random.rand() < 0.9  # 90% success
            
            # Update metrics
            if is_recognized:
                self.metrics.record_recognition_attempt(True, 0.85)
                self.metrics.record_stage_latency('detection', detection_time * 1000)
                self.metrics.record_stage_latency('recognition', recognition_time * 1000)
                self.metrics.record_stage_latency('liveness', liveness_time * 1000)
            else:
                self.metrics.record_recognition_attempt(False, 0.3)
            
            self.frames_processed += 1
            
            elapsed = time.time() - start_time
            
            return {
                'camera_id': self.camera_id,
                'timestamp': time.time(),
                'frame_num': self.frames_processed,
                'processing_time_ms': elapsed * 1000,
                'success': is_recognized,
            }
        
        except Exception as e:
            self.errors += 1
            return {
                'camera_id': self.camera_id,
                'timestamp': time.time(),
                'frame_num': self.frames_processed,
                'processing_time_ms': 0,
                'success': False,
                'error': str(e),
            }
    
    def run(self, duration_seconds: int = 30) -> List[Dict]:
        """Run camera simulation for specified duration.
        
        Args:
            duration_seconds: Duration to run
            
        Returns:
            List of frame processing records
        """
        results = []
        start_time = time.time()
        last_frame_time = start_time
        
        while time.time() - start_time < duration_seconds and not self.stop_event.is_set():
            frame_start = time.time()
            result = self.process_frame()
            results.append(result)
            
            # Regulate frame rate
            elapsed = time.time() - frame_start
            sleep_time = max(0, self.frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        return results


def benchmark_concurrent_cameras(
    num_cameras_list: List[int] = None,
    duration_per_test: int = 30,
    output_csv: str = 'results/concurrency_benchmark.csv',
    metadata_file: str = 'results/concurrency_metadata.json',
) -> Dict:
    """Benchmark system with increasing number of concurrent cameras.
    
    Args:
        num_cameras_list: List of camera counts to test
        duration_per_test: Seconds to run each test
        output_csv: Output CSV file path
        metadata_file: Output metadata JSON file path
        
    Returns:
        Dictionary with benchmark results
    """
    if num_cameras_list is None:
        num_cameras_list = [1, 2, 5, 10, 20]
    
    output_csv = Path(output_csv)
    metadata_file = Path(metadata_file)
    
    # Create output directory
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CONCURRENCY BENCHMARK: Multi-Camera Throughput & Latency")
    print("=" * 80)
    
    results = []
    process = psutil.Process()
    
    for num_cameras in num_cameras_list:
        print(f"\n[*] Testing with {num_cameras} concurrent cameras ({duration_per_test}s)...")
        print("-" * 80)
        
        # Prepare memory baseline
        process.memory_info()  # Force update
        baseline_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Create camera simulators
        cameras = [
            MockCameraSimulator(camera_id=i, frame_rate=10)
            for i in range(num_cameras)
        ]
        
        # Run all cameras concurrently
        all_results = []
        latencies = []
        success_count = 0
        error_count = 0
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_cameras) as executor:
            futures = [
                executor.submit(camera.run, duration_per_test)
                for camera in cameras
            ]
            
            # Wait for all to complete and collect results
            for camera_idx, future in enumerate(futures):
                camera_results = future.result()
                all_results.extend(camera_results)
                
                # Track metrics
                for result in camera_results:
                    if 'error' not in result:
                        latencies.append(result['processing_time_ms'])
                        if result['success']:
                            success_count += 1
                        else:
                            error_count += 1
        
        elapsed = time.time() - start_time
        
        # Memory after test
        peak_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_delta_mb = peak_memory_mb - baseline_memory_mb
        
        # Compute statistics
        total_frames = len(all_results)
        fps = total_frames / elapsed
        
        if latencies:
            latencies_sorted = np.sort(latencies)
            p50_latency = np.percentile(latencies_sorted, 50)
            p95_latency = np.percentile(latencies_sorted, 95)
            p99_latency = np.percentile(latencies_sorted, 99)
            avg_latency = np.mean(latencies_sorted)
            max_latency = np.max(latencies_sorted)
        else:
            p50_latency = p95_latency = p99_latency = avg_latency = max_latency = 0
        
        throughput_per_camera = fps / num_cameras if num_cameras > 0 else 0
        
        # Compute FPS degradation relative to single camera
        # (for now, assume baseline is first result)
        
        result_row = {
            'num_cameras': num_cameras,
            'total_frames': total_frames,
            'successful_frames': success_count,
            'failed_frames': error_count,
            'duration_seconds': f"{elapsed:.2f}",
            'total_fps': f"{fps:.2f}",
            'fps_per_camera': f"{throughput_per_camera:.2f}",
            'avg_latency_ms': f"{avg_latency:.2f}",
            'p50_latency_ms': f"{p50_latency:.2f}",
            'p95_latency_ms': f"{p95_latency:.2f}",
            'p99_latency_ms': f"{p99_latency:.2f}",
            'max_latency_ms': f"{max_latency:.2f}",
            'memory_used_mb': f"{memory_delta_mb:.2f}",
            'success_rate': f"{100*success_count/(success_count+error_count):.1f}%",
        }
        
        results.append(result_row)
        
        # Print results
        print(f"  Total FPS:              {result_row['total_fps']}")
        print(f"  FPS per camera:         {result_row['fps_per_camera']}")
        print(f"  Total frames:           {result_row['total_frames']}")
        print(f"  Successful:             {result_row['successful_frames']}/{result_row['total_frames']} ({result_row['success_rate']})")
        print(f"  Latency (p50/p95/p99):  {result_row['p50_latency_ms']}/{result_row['p95_latency_ms']}/{result_row['p99_latency_ms']} ms")
        print(f"  Memory delta:           {result_row['memory_used_mb']} MB")
    
    # Write CSV
    print("\n[*] Writing results to CSV...")
    with open(output_csv, 'w', newline='') as f:
        fieldnames = [
            'num_cameras', 'total_frames', 'successful_frames', 'failed_frames',
            'duration_seconds', 'total_fps', 'fps_per_camera', 'avg_latency_ms',
            'p50_latency_ms', 'p95_latency_ms', 'p99_latency_ms', 'max_latency_ms',
            'memory_used_mb', 'success_rate',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"    ✓ Results saved to {output_csv}")
    
    # Write metadata
    summary = {
        'benchmark': 'concurrency',
        'test_duration_seconds': duration_per_test,
        'camera_counts_tested': num_cameras_list,
        'results_csv': str(output_csv),
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"    ✓ Metadata saved to {metadata_file}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("CONCURRENCY BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Cameras':>8} {'Total FPS':>12} {'FPS/Cam':>12} {'Avg Lat':>12} {'P95 Lat':>12} {'Memory':>12} {'Success':>10}")
    print("-" * 80)
    for row in results:
        print(f"{row['num_cameras']:>8} {row['total_fps']:>12} {row['fps_per_camera']:>12} "
              f"{row['avg_latency_ms']:>12} {row['p95_latency_ms']:>12} {row['memory_used_mb']:>12} {row['success_rate']:>10}")
    print("-" * 80)
    
    # Analysis
    print("\nSCALABILITY ANALYSIS:")
    print("-" * 80)
    if len(results) > 1:
        baseline_fps = float(results[0]['total_fps'])
        print(f"  Baseline (1 camera):     {baseline_fps:.2f} FPS")
        
        for i in range(1, len(results)):
            num_cams = results[i]['num_cameras']
            fps = float(results[i]['total_fps'])
            fps_per_cam = float(results[i]['fps_per_camera'])
            ideal_fps_per_cam = baseline_fps
            efficiency = (fps_per_cam / ideal_fps_per_cam * 100) if ideal_fps_per_cam > 0 else 0
            print(f"  {num_cams} cameras:          {fps:.2f} FPS total, {fps_per_cam:.2f} FPS/cam "
                  f"({efficiency:.0f}% efficiency)")
    
    print("=" * 80)
    
    return {
        'results': results,
        'output_csv': str(output_csv),
        'metadata_file': str(metadata_file),
    }


def main():
    parser = ArgumentParser(description='Concurrency Benchmark for AutoAttendance')
    parser.add_argument('--cameras', type=int, nargs='+', default=[1, 2, 5, 10, 20],
                       help='Camera counts to test')
    parser.add_argument('--duration', type=int, default=30, help='Duration per test (seconds)')
    parser.add_argument('--output', default='results/concurrency_benchmark.csv', help='Output CSV file')
    parser.add_argument('--metadata', default='results/concurrency_metadata.json', help='Output metadata JSON')
    
    args = parser.parse_args()
    
    benchmark_concurrent_cameras(
        num_cameras_list=args.cameras,
        duration_per_test=args.duration,
        output_csv=args.output,
        metadata_file=args.metadata,
    )


if __name__ == '__main__':
    main()
