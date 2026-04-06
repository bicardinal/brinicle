import json
import re
from bs4 import BeautifulSoup

special_chars = json.load(open("examples/special_chars.json"))["vocab"]

def clean_space(text):
	text = text.replace("\u200c", " ").replace("\u200b", "")
	text = re.sub(r"[ ]+", " ", text)
	text = re.sub(r"[\n]+", "\n", text)
	text = re.sub(r"[\t]+", "\t", text)
	return text.strip()

def clean_special_char(text):
	for k, v in special_chars.items():
		text = text.replace(k, v)
	return text

def clean_nums(text):
	# text = re.sub(r"[\d]+", "<N>", text)
	return text

def html_clean(text):
	if isinstance(text, str):
		text = clean_special_char(text)
		if "<" not in text:
			text = clean_nums(text)
			return clean_space(text).strip().lower()
		text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
		text = clean_space(text)
		text = clean_nums(text)
		return text.strip().lower()
	return ""