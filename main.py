import os
import re
import csv
import glob

from enum import StrEnum
from typing import Iterable
from pydantic import BaseModel
from datetime import datetime, date, time, timedelta
from dateutil import parser

import cv2
import pytesseract
import numpy as np

from functools import partial, reduce
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


# Ensure pytesseract can find the Tesseract-OCR executable
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


hour_translation = {'eng': 'hr', 'rus': 'ч'}
minute_translation = {'eng': 'min', 'rus': 'мин'}
second_translation = {'eng': 'sec', 'rus': 'с'}


class CallType(StrEnum):
    AUDIO = 'audio'
    VIDEO = 'video'


class CallDirection(StrEnum):
    IN = 'incoming'
    OUT = 'outgoing'


# @dataclass
class Call(BaseModel):
    type: CallType = None
    direction: CallDirection = None
    missed: bool = None

    timestamp: datetime | None = None
    duration: timedelta | None = None


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


def call_parser(pt: tuple[int, int], w: int, h: int, image: cv2.typing.MatLike,
                call_type: CallType, lang: str) -> tuple[int, Call]:
    image_center = image.shape[0] / 2

    call_image = crop_image(image, pt[0], pt[1], w, h)
    call_time, call_duration = image_to_time(call_image, lang)

    return pt[1], Call(
        direction = CallDirection.IN if pt[0] < image_center else CallDirection.OUT,
        type = call_type,
        missed = call_duration is None,
        timestamp = datetime.combine(date(1, 1, 1), call_time) if call_time else None,
        duration = call_duration,
    )


def screenshot_to_calls(screenshot_path: str, lang: str) -> list[Call]:
    image = preprocess_image(screenshot_path)

    audio_calls, w, h = template_matching(image, 'resources/audio.png')
    video_calls, _, _ = template_matching(image, 'resources/video.png')

    processes = min(os.cpu_count(), max(len(audio_calls), len(video_calls)))
    custom_call_parser = partial(call_parser, w=w, h=h, image=image, lang=lang)

    with ThreadPool(processes) as pool:
        _, call_list = zip(*sorted(
            pool.map(partial(custom_call_parser, call_type=CallType.AUDIO), audio_calls) +
            pool.map(partial(custom_call_parser, call_type=CallType.VIDEO), video_calls)
        ))
        return call_list


def merge_call_lists(previous: list[Call], next: list[Call]) -> list[Call]:
    for i in range(1, 1 + min(len(previous), len(next))):
        if previous[-i:] == next[:i]:
            return previous + next[i:]

    return previous + next


def export_to_csv(path: str, calls: list[Call]):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=';', fieldnames=Call.model_fields.keys())
        writer.writeheader()
        for call in calls:
            writer.writerow(call.model_dump())


def import_from_csv(path: str) -> Iterable[Call]:
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            yield Call(**{k: v for k, v in row.items() if v})


def import_from_txt(path: str) -> Iterable[Call]:
    with open(path, encoding='utf-8') as file:
        for line in file:
            match = re.search(r'^(?P<timestamp>.*)\s+\-\s+(?P<author>.*)\: null$', line)
            if match:
                yield Call(
                    timestamp = parser.parse(match.group('timestamp')),
                    direction = CallDirection.IN if match.group('author') in path \
                        else CallDirection.OUT,
                )


def expand_calls_by_chat(calls: list[Call], chat_nulls: list[Call]) -> list[Call]:
    in_out_mode = {call.direction for call in calls} == {call.direction for call in chat_nulls}

    i, j = 1, 1
    while i <= len(calls) and j <= len(chat_nulls):
        if (calls[-i].timestamp is None or calls[-i].timestamp.time() == chat_nulls[-j].timestamp.time()) and \
                (not in_out_mode or calls[-i].direction == chat_nulls[-j].direction):
            calls[-i].timestamp = chat_nulls[-j].timestamp
            i = i + 1

        j = j + 1

    return calls


if __name__ == '__main__':
    lang = 'rus'
    screenshots = sorted(glob.glob('Screenshot_*.jpg'))

    # call_lists = []
    # with Pool(min(os.cpu_count(), len(screenshots))) as pool:
    #     call_lists = pool.map(partial(screenshot_to_calls, lang=lang), screenshots)

    # calls = reduce(merge_call_lists, call_lists)
    # export_to_csv('calls.csv', calls)

    calls = import_from_csv('calls.csv')
    export_to_csv('calls2.csv', calls)

    # chat = glob.glob('*WhatsApp*.txt')[0]
    # chat_nulls = import_from_txt(chat)
    # # export_to_csv('chat.csv', chat_nulls)
    # calls = expand_calls_by_chat(list(calls), list(chat_nulls))
    # export_to_csv('expanded_calls.csv', calls)
