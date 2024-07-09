import os
import re
import csv
import glob
import difflib

from enum import StrEnum
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
    x_image_right_border = image.shape[1]

    return image[y_template_center:y_template_lower_border,
                 x_template_right_border:x_image_right_border]


def text_to_time(text: str, lang: str) -> tuple[time, timedelta]:
    lines = text.splitlines()
    duration_match = re.search(r'^\s*(?P<hours>\d{1,2})\D+(?P<minutes>\d{1,2})', lines[0])

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
    image_center = image.shape[1] / 2

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
    while not previous[-1].timestamp:
        previous = previous[:-1]

    for i in range(1, 1 + min(len(previous), len(next)))[::-1]:
        if previous[-i:] == next[:i]:
            return previous + next[i:]

    return previous + next


def export_to_csv(path: str, calls: list[Call]):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=';', fieldnames=Call.model_fields.keys())
        writer.writeheader()
        for call in calls:
            writer.writerow(call.model_dump())


def import_from_csv(path: str) -> list[Call]:
    result = []
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            result.append(Call(**{k: v for k, v in row.items() if v}))

    return result


def import_from_txt(path: str) -> list[Call]:
    result = []
    with open(path, encoding='utf-8') as file:
        for line in file:
            match = re.search(r'^(?P<timestamp>.*)\s+\-\s+(?P<author>.*)\: null$', line)
            if match:
                result.append(Call(
                    timestamp = parser.parse(match.group('timestamp')),
                    direction = CallDirection.IN if match.group('author') in path \
                        else CallDirection.OUT,
                ))

    return result


def compare_time(lhs: datetime, rhs: datetime, epsilon: timedelta = timedelta(minutes=1)) -> bool:
    delta = abs(rhs - lhs.replace(year=rhs.year, month=rhs.month, day=rhs.day))
    return delta <= epsilon or (timedelta(days=1) - delta) < epsilon


def recognition_correction(calls: list[Call], nulls: list[Call]) -> list[Call]:
    assert len(calls) <= len(nulls)
    if len(calls) == 0:
        return []

    if len(calls) == len(nulls):
        for i in range(len(calls)):
            calls[i].timestamp = nulls[i].timestamp
    else:
        print()
        for i in range(len(nulls)):
            if i < len(calls):
                print(f'{nulls[i].direction} {nulls[i].timestamp} {calls[i].direction} {calls[i].timestamp}')
            else:
                print(f'{nulls[i].direction} {nulls[i].timestamp}')
        print()

    return calls


def expand_calls_by_chat(calls: list[Call], nulls: list[Call]) -> list[Call]:
    assert len(calls) <= len(nulls)

    in_out_mode = {call.direction for call in calls} == {call.direction for call in nulls}
    comparator = lambda call_index, null_index: (calls[call_index].timestamp is None or
        compare_time(calls[call_index].timestamp, nulls[null_index].timestamp), timedelta(minutes=1)) and \
            (not in_out_mode or calls[call_index].direction == nulls[null_index].direction)

    calls, nulls = calls[::-1], nulls[::-1]
    call_lower_index = call_upper_index = 0
    null_lower_index = null_upper_index = 0

    while call_lower_index < len(calls) and call_upper_index < len(calls) and \
            null_lower_index < len(nulls) and null_upper_index < len(nulls):
        call_upper_index = call_upper_index + 1

        for i in range(call_lower_index, call_upper_index):
            assert i - call_lower_index <= null_upper_index - null_lower_index
            if comparator(i, null_upper_index):
                calls[i].timestamp = nulls[null_upper_index].timestamp
                calls[call_lower_index:i] = recognition_correction(calls[call_lower_index:i],
                                                                   nulls[null_lower_index:null_upper_index])
                null_lower_index = null_upper_index + 1
                call_lower_index = call_upper_index = i + 1
                break

        null_upper_index = null_upper_index + 1

    print(f'calls[{call_lower_index}:{call_upper_index}], nulls[{null_lower_index}:{null_upper_index}]')
    print(f'calls[{call_lower_index}:{len(calls)}], nulls[{null_lower_index}:{len(nulls)}]')
    calls[call_lower_index:call_upper_index] = recognition_correction(calls[call_lower_index:call_upper_index],
                                                                      nulls[null_lower_index:null_upper_index])
    return calls[::-1]


def expand_calls_by_chat_quadratic(calls: list[Call], nulls: list[Call]) -> list[Call]:
    assert len(calls) <= len(nulls)

    in_out_mode = {call.direction for call in calls} == {call.direction for call in nulls}
    comparator = lambda call_index, null_index: (calls[call_index].timestamp is None or
        compare_time(calls[call_index].timestamp, nulls[null_index].timestamp)) and \
            (not in_out_mode or calls[call_index].direction == nulls[null_index].direction)

    calls, nulls = calls[::-1], nulls[::-1]
    lower_index = 0

    for i in range(len(calls)):
        old_lower_index = lower_index
        upper_index = min(lower_index + len(nulls[lower_index:]) - len(calls[i:]) + 1, len(nulls))
        for j in range(lower_index, upper_index):
            if comparator(i, j):
                calls[i].timestamp = nulls[j].timestamp
                lower_index = j + 1
                break

        if old_lower_index == lower_index:
            print(calls[i].timestamp)
            for call in nulls[lower_index:upper_index]:
                print(f'    {call.timestamp}')

    return calls[::-1]


if __name__ == '__main__':
    lang = 'rus'
    screenshots = sorted(glob.glob('Screenshot_*.jpg'))

    # call_lists = []
    # with Pool(min(os.cpu_count(), len(screenshots))) as pool:
    #     call_lists = pool.map(partial(screenshot_to_calls, lang=lang), screenshots)

    # calls = reduce(merge_call_lists, call_lists)
    # export_to_csv('calls.csv', calls)
    calls = import_from_csv('calls.csv')

    chat = glob.glob('*WhatsApp*.txt')[0]
    chat_nulls = import_from_txt(chat)
    export_to_csv('chat.csv', chat_nulls)

    calls = expand_calls_by_chat_quadratic(calls, chat_nulls)
    export_to_csv('expanded_calls.csv', calls)


    # import difflib
    # # from Levenshtein import distance as levenshtein_distance

    # eng = glob.glob('*WhatsApp*.txt')[0]
    # rus = glob.glob('*WhatsApp*.txt')[1]
    # example = eng
    # fst = 'Елизавета Выборнова'
    # snd = 'Виктор Фадеев'

    # # print(f"'{os.path.basename(eng)}' '{os.path.basename(rus)}' '{fst}' '{snd}' ")
    # # print(os.path.splitext('t.t.t'))

    # example = os.path.splitext(example)[0]
    # example_fst = example[-len(fst):]
    # example_snd = example[-len(snd):]
    # print(f"'{example_fst}' '{example_snd}'")
    # fst_ratio = difflib.SequenceMatcher(None, example_fst, fst).ratio()
    # snd_ratio = difflib.SequenceMatcher(None, example_snd, snd).ratio()
    # print(f'{fst_ratio} {snd_ratio}')
    # print(difflib.SequenceMatcher(None, '2453', '123').ratio())

