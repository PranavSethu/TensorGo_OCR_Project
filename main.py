import os
import cv2
from src.ocr_processor import create_reader
from src.performance_evaluator import evaluate_performance, compare_performances

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    image_folder = os.path.join(base_path, 'data', 'images')
    image_paths = [os.path.join(image_folder, f'frame{i}.png') for i in range(1, 4)]
    images = [cv2.imread(path) for path in image_paths if os.path.exists(path)]

    gpu_reader = create_reader(gpu=True)
    cpu_reader = create_reader(gpu=False)

    gpu_texts, gpu_fps, gpu_accuracy = evaluate_performance(gpu_reader, images, use_gpu=True)
    cpu_texts, cpu_fps, cpu_accuracy = evaluate_performance(cpu_reader, images, use_gpu=False)

    compare_performances((gpu_texts, gpu_fps, gpu_accuracy), (cpu_texts, cpu_fps, cpu_accuracy))

if __name__ == '__main__':
    main()

