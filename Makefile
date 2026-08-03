.PHONY: reproduce test check

PYTHON ?= .venv/bin/python

reproduce:
	bash scripts/reproduce.sh

test:
	$(PYTHON) -m unittest discover -s tests -v

check: reproduce test
