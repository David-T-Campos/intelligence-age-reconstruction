PYTHON := .venv/bin/python

.PHONY: setup extract render compare inspect review isolated audio test smoke all
setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt
extract:
	$(PYTHON) scripts/pipeline.py extract
render:
	$(PYTHON) scripts/pipeline.py render
compare:
	$(PYTHON) scripts/pipeline.py compare
inspect:
	$(PYTHON) scripts/pipeline.py inspect --frame 168
review:
	$(PYTHON) scripts/make_review.py
isolated:
	$(PYTHON) scripts/verify_isolation.py
audio:
	$(PYTHON) scripts/check_audio.py
test:
	$(PYTHON) -m unittest discover -s tests -v
smoke:
	$(PYTHON) scripts/smoke_test.py
all:
	$(MAKE) extract
	$(MAKE) render
	$(MAKE) compare
	$(MAKE) audio
