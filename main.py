import cv2
import pytesseract
import numpy as np
import glob
import os
import re
from datetime import datetime, time, timedelta
from enum import Enum

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



hour_translation = {'eng': 'hr', 'rus': 'ч'}
minute_translation = {'eng': 'min', 'rus': 'мин'}
second_translation = {'eng': 'sec', 'rus': 'с'}


class CallType(Enum):
    AUDIO = 'audio'
    VIDEO = 'video'


class CallDirection(Enum):
    IN = 'incoming'
    OUT = 'outgoing'


class Call():
    type: CallType
    direction: CallDirection
    missed: bool
    duration: timedelta
    timestamp: datetime


def preprocess_image(image_path: str) -> cv2.typing.MatLike:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary_image


def template_matching(image: cv2.typing.MatLike, template_path: str) -> tuple[list[tuple[int, int]], int, int]:
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    width, height = template.shape[::-1]

    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)

    filtered = []
    for pt in zip(*loc[::-1]):
        if all([abs(pt[0] - fp[0]) > width or abs(pt[1] - fp[1]) > height for fp in filtered]):
            filtered.append(pt)

    return filtered, width, height


def crop_image(image: cv2.typing.MatLike, x: int, y: int, width: int, height: int) -> cv2.typing.MatLike:
    y_image_center = y + int(height / 2)
    y_lower_border = y + height
    x_right_border = x + width

    return image[y_image_center:y_lower_border, x_right_border:width]


def text_to_time(text: str, lang: str) -> tuple[time, timedelta]:
    lines = text.splitlines()
    duration_match = re.search(r'^\s*(?P<hours>\d{1,2})\D*(?P<minutes>\d{1,2})', lines[0])

    hour_character = hour_translation[lang][0]
    hours_match = re.search(r'^\s*(?P<hours>\d{1,2})\s*' + hour_character, lines[0])

    minute_character = minute_translation[lang][0]
    minutes_match = re.search(r'^\s*(?P<minutes>\d{1,2})\s*' + minute_character, lines[0])

    second_character = second_translation[lang][0]
    seconds_match = re.search(r'^\s*(?P<seconds>\d{1,2})\s*' + second_character, lines[0])

    duration = timedelta(
        hours=int(duration_match.group('hours')),
        minutes=int(duration_match.group('minutes')),
    ) if duration_match else timedelta(
        hours=int(hours_match.group('hours')),
    ) if hours_match else timedelta(
        minutes=int(minutes_match.group('minutes')),
    ) if minutes_match else timedelta(
        seconds=int(seconds_match.group('seconds')),
    ) if seconds_match else None

    time_match = re.search(r'(^|\D)(?P<hour>\d{2}).?(?P<minute>\d{2})\s*$', lines[-1])
    timestamp = None if not time_match else time(
        hour=int(time_match.group('hour')),
        minute=int(time_match.group('minute')),
    )

    return timestamp, duration


def image_to_time(image: cv2.typing.MatLike, lang: str) -> tuple[time, timedelta]:
    config = os.path.abspath(f'resources/{lang}.config')
    call_text = pytesseract.image_to_string(call_image, lang=lang, config=f'"{config}"')
    return text_to_time(call_text, lang)


if __name__ == '__main__':
    lang = 'rus'
    screenshots = glob.glob('Screenshot_*.jpg')
    screenshots = [screenshots[0]]

    for screenshot in screenshots:
        image = preprocess_image(screenshot)
        width, height = image.shape[::-1]

        audio_calls, w, h = template_matching(image, 'resources/audio.bmp')
        video_calls, _, _ = template_matching(image, 'resources/video.bmp')

        # log = []
        # file = open('log.txt', 'w', encoding='utf-8')
        for x, y in audio_calls:
            call_image = crop_image(image, x, y, w, h)
            call_time, call_duration = image_to_time(call_image, lang)



        #     if call_time is None or duration is None:
        #         file.write(f'time = {call_time}, duration = {duration}, text = "{text}"\n')
        #     else:
        #         file.write(f'time = {call_time}, duration = {duration}\n')

        # file.close()
