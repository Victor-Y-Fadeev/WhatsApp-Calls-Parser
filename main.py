import os
import re
import glob
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, time, timedelta

import cv2
import pytesseract
import numpy as np


# Ensure pytesseract can find the Tesseract-OCR executable
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


hour_translation = {'eng': 'hr', 'rus': 'ч'}
minute_translation = {'eng': 'min', 'rus': 'мин'}
second_translation = {'eng': 'sec', 'rus': 'с'}


class CallType(Enum):
    AUDIO = 'audio'
    VIDEO = 'video'


class CallDirection(Enum):
    IN = 'incoming'
    OUT = 'outgoing'


@dataclass
class Call:
    type: CallType
    direction: CallDirection
    missed: bool

    timestamp: datetime
    duration: timedelta


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
    y_template_center = y + int(height / 2)
    y_template_lower_border = y + height
    x_template_right_border = x + width
    x_image_right_border = image.shape[0]

    return image[y_template_center:y_template_lower_border,
                 x_template_right_border:x_image_right_border]


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
    call_text = pytesseract.image_to_string(image, lang=lang, config=f'"{config}"')
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

        file = open('log.txt', 'w', encoding='utf-8')
        for x, y in audio_calls:
            call_image = crop_image(image, x, y, w, h)
            call_time, call_duration = image_to_time(call_image, lang)
            call = Call(
                type = CallType.AUDIO,
                direction = CallDirection.IN if x < width / 2 else CallDirection.OUT,
                missed = call_duration is None,
                timestamp = call_time,
                duration = call_duration,
            )

            file.write(f'{call.type.value} {call.direction.value} {call.missed} {call.timestamp} {call.duration}\n')

        file.close()
