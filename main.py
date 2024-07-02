import cv2
import pytesseract
import numpy as np
import glob
import os

# Ensure pytesseract can find the Tesseract-OCR executable
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


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


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary_image

def template_matching(image, template_path):
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    width, height = template.shape[::-1]

    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)

    return zip(*loc[::-1]), width, height

if __name__ == '__main__':
    screenshots = glob.glob('Screenshot_*.jpg')
    screenshots = [screenshots[0]]

    # cv2.imwrite('threshold.png', preprocess_image(screenshots[0]))
    # exit()
    for screenshot in screenshots:
        image = preprocess_image(screenshot)
        width, height = image.shape[::-1]

        audio_calls, w, h = template_matching(image, 'resources/audio.bmp')
        video_calls, _, _ = template_matching(image, 'resources/video.bmp')

        # log = []
        file = open('log.txt', 'w', encoding='utf-8')
        for x, y in audio_calls:
            cut = image[y + int(h / 2):y + h, x + w:width]


            # text = pytesseract.image_to_string(cut, config=f'"{path}"')
            # patterns = os.path.abspath('my.patterns')
            config = os.path.abspath('resources/rus.user-config')
            # patterns = os.path.abspath('resources/rus.user-patterns')
            # words = os.path.abspath('resources/rus.user-words')
            # text = pytesseract.image_to_string(cut, lang='rus', config=f'"{config}" --user-words "{words}" --user-patterns "{patterns}"')
            text = pytesseract.image_to_string(cut, lang='rus', config=f'"{config}"')
            # log.append(text)
            # print(text)
            file.write(f'"{text}"')
            if '251ро' in text:
                cv2.imwrite('res.png', cut)


        file.close()

    # image = preprocess_image('Arial.png')
    # config = os.path.abspath('resources/russian.config')
    # text = pytesseract.image_to_string(image, lang='eng', config=f'"{config}"')
    # print(f'"{text}"')



        # path = os.path.abspath('resources/russian.config')
        # print(path)

        # patterns = os.path.abspath('resources/call.patterns')