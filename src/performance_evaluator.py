import time
import matplotlib.pyplot as plt
from src.ocr_processor import preprocess_image, create_reader

def calculate_accuracy(detected_texts, ground_truths):
    """Calculate the accuracy based on the comparison between detected texts and ground truths."""
    correct = 0
    for detected, truth in zip(detected_texts, ground_truths):
        if detected.strip().lower() == truth.strip().lower():
            correct += 1
    return correct / len(ground_truths) if ground_truths else 0

def evaluate_performance(reader, images, use_gpu, ground_truths=None):
    start_time = time.time()
    detected_texts = []
    for img in images:
        preprocessed_img = preprocess_image(img, for_cpu=not use_gpu)
        results = reader.readtext(preprocessed_img)
        detected_texts.extend([result[1] for result in results])  
    end_time = time.time()
    fps = len(images) / (end_time - start_time)
    accuracy = calculate_accuracy(detected_texts, ground_truths) if ground_truths else 0 
    return len(detected_texts), fps, accuracy

def compare_performances(gpu_data, cpu_data):
    labels = ['GPU', 'CPU']
    fps_values = [gpu_data[1], cpu_data[1]]
    accuracy_values = [gpu_data[2] if gpu_data[2] is not None else 0, cpu_data[2] if cpu_data[2] is not None else 0]  

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, fps_values, width, label='FPS')
    rects2 = ax.bar(x, accuracy_values, width, bottom=fps_values, label='Accuracy')

    ax.set_ylabel('Scores')
    ax.set_title('Performance comparison between GPU and CPU')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.show()
    fig.savefig('results/performance_comparison_chart.png')

