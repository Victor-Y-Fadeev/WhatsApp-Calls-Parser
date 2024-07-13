import os
import re
import csv
import glob

from bs4 import BeautifulSoup, Tag


def message_filter(div: Tag) -> bool:
	body = div.select_one('.body')
	if not body:
		return False

	classes = list(map(lambda div: div.get('class'), body.find_all()))
	return classes == [['title', 'bold'], ['status', 'details']]


if __name__ == '__main__':
	root = os.path.dirname(__file__)
	paths = map(lambda path: os.path.join(root, path), glob.glob('messages*.html', root_dir=root))

	for html in paths:
		with open(html, encoding='utf-8') as fp:
			soup = BeautifulSoup(fp)
			messages = soup.select('.message.default')

			bodies = map(lambda div: div.select_one('.body'), messages)
			bodies = list(bodies)

			filtered = filter(message_filter, bodies)
			filtered = list(filtered)


			pairs = map(lambda body: (body.select_one('.date')['title'], body.select_one('.status').string), filtered)
			pairs = list(pairs)

			print(pairs[0])


			# classes = lambda body: list(map(lambda div: div['class'], body.find_all()))
			# filtered = filter(lambda body: map(lambda div: div['class'], body.select_one('.body').find_all()) == [['title', 'bold'], ['status', 'details']], bodies)


			# print(len(list(filtered)))
			# print(len(pairs))
			# print(classes(pairs[0][1]))



			# print(messages[0].select_one('.body'))
			# print([div['class'] for div in messages[0]])

			# print(len(messages[0].find_all(recursive=False)))
			# print(messages[0].select_one('.pull_right.date.details')['title'])
			# print(messages[0].select_one('.body').select_one('.pull_right.date.details')['title'])

			# print(len(list(messages[0].contents)))
			# print(len(list(messages[0].select('div.body')[0].contents)))



			# # print(len(messages))
			# message = soup.find('div', class_='message default clearfix')
			# # print(message.find('div', class_='body').prettify())
			# print(message.class_)