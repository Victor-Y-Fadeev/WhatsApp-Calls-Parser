import os
import re
import csv
import glob

from enum import StrEnum
from pydantic import BaseModel
from datetime import datetime, timedelta
from dateutil import parser

from bs4 import BeautifulSoup, Tag
from functools import reduce


class CallType(StrEnum):
	INCOMING = 'Incoming'
	MISSED = 'Missed'
	DECLINED = 'Declined'

	OUTGOING = 'Outgoing'
	CANCELLED ='Cancelled'


class Call(BaseModel):
    type: CallType
    timestamp: datetime
    duration: timedelta | None


def message_filter(div: Tag) -> bool:
	body = div.select_one('.body')
	if not body:
		return False

	classes = list(map(lambda div: div.get('class'), body.find_all()))
	return classes == [['title', 'bold'], ['status', 'details']]


def import_from_html(path: str) -> list[tuple[str, str]]:
	with open(path, encoding='utf-8') as fp:
		soup = BeautifulSoup(fp, 'html.parser')

		messages = soup.select('.message.default')
		bodies = map(lambda div: div.select_one('.body'), messages)
		filtered = filter(message_filter, bodies)

		pairs = map(lambda body: (body.select_one('.date')['title'], body.select_one('.status').string), filtered)
		return list(pairs)


def div_to_call(item: tuple[str, str]) -> Call:
	match = re.search(r'^\s*(?P<type>\w+)(\s*\((?P<seconds>\d+)\D*\))?\s*$', item[1])
	duration = match.group('seconds')

	return Call(
		type=CallType(match.group('type')),
		timestamp=parser.parse(item[0], dayfirst='/' not in item[0], tzinfos={'MSK': 3 * 3600}),
		duration=timedelta(seconds=int(duration)) if duration else None
	)


def export_to_csv(path: str, calls: list[Call]):
    with open(path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=';', fieldnames=Call.model_fields.keys())
        writer.writeheader()
        for call in calls:
            writer.writerow(call.model_dump())


if __name__ == '__main__':
	root = os.path.dirname(__file__)
	paths = map(lambda path: os.path.join(root, path), glob.glob('messages*.html', root_dir=root))

	imported = reduce(lambda lhs, rhs: [*lhs, *rhs], map(import_from_html, paths))
	calls = list(map(div_to_call, imported))

	export_to_csv(os.path.join(root, 'telegram.csv'), calls)
