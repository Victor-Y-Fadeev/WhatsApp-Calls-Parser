import cv2
import pytesseract
from PIL import Image
import numpy as np

# Ensure pytesseract can find the Tesseract-OCR executable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Read the image in grayscale
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # Thresholding to binary image
    return binary_image

def template_matching(image, template_path):
    template = cv2.imread(template_path, 0) # Read template in grayscale
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    points = []
    for pt in zip(*loc[::-1]):
        points.append(pt)
    return points

def extract_text(image):
    text = pytesseract.image_to_string(image)
    return text

def main(screenshot_path, incoming_icon_path, outgoing_icon_path):
    processed_image = preprocess_image(screenshot_path)

    incoming_calls = template_matching(processed_image, incoming_icon_path)
    outgoing_calls = template_matching(processed_image, outgoing_icon_path)

    print(f"Incoming call icons found at: {incoming_calls}")
    print(f"Outgoing call icons found at: {outgoing_calls}")

    # Assuming call details are in the same bounding boxes as icons; adjust as necessary
    text = extract_text(processed_image)
#     print(f"Extracted Text: {text}")

# if name == "main":
#     screenshot_path = 'your_screenshot.png'
#     incoming_icon_path = 'incoming_icon_template.png'
#     outgoing_icon_path = 'outgoing_icon_template.png'
#     main(screenshot_path, incoming_icon_path, outgoing_icon_path)





# image = cv2.imread('Screenshot.png')
# img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.imread('Screenshot.png', cv2.IMREAD_GRAYSCALE)
width, height = image.shape[::-1]
_, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


# import cv2 as cv
# img_rgb = cv.imread('Screenshot.png')
# assert img_rgb is not None, "file could not be read, check with os.path.exists()"
# img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv2.imread('audio.bmp', cv2.IMREAD_GRAYSCALE)
assert template is not None, "file could not be read, check with os.path.exists()"
w, h = template.shape[::-1]

res = cv2.matchTemplate(thresh, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(thresh, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imwrite('res.png', thresh)
# cv2.imshow('Binary Threshold', img_rgb)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()