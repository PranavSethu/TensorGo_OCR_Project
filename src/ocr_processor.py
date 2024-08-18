import cv2
import easyocr

def create_reader(gpu=True):
    """Create an OCR reader either for GPU or CPU."""
    return easyocr.Reader(['en'], gpu=gpu)


def preprocess_image(img, for_cpu=False):
    """Apply preprocessing to enhance image for OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    if for_cpu:

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return thresh

