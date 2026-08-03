.PHONY: reproduce test verify-mapping-strength check

PYTHON ?= .venv/bin/python

reproduce:
	bash scripts/reproduce.sh

test:
	$(PYTHON) -m unittest discover -s tests -v

verify-mapping-strength:
	$(PYTHON) scripts/verify_mapping_strength_archive.py

check: reproduce test verify-mapping-strength
