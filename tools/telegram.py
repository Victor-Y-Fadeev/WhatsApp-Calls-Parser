import os
import re
import csv
import glob

from bs4 import BeautifulSoup, Tag
from functools import reduce


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


if __name__ == '__main__':
	root = os.path.dirname(__file__)
	paths = map(lambda path: os.path.join(root, path), glob.glob('messages*.html', root_dir=root))

	calls = reduce(lambda lhs, rhs: [*lhs, *rhs], map(import_from_html, paths))
	print(len(calls))