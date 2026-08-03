.PHONY: test verify-mapping-strength check

PYTHON ?= .venv/bin/python

test:
	$(PYTHON) -m unittest discover -s tests -v

verify-mapping-strength:
	$(PYTHON) scripts/verify_mapping_strength_archive.py

check: test verify-mapping-strength
