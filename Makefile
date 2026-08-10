.PHONY: test integrated-best verify-mapping-strength check

PYTHON ?= .venv/bin/python

test:
	$(PYTHON) -m unittest discover -s tests -v

integrated-best:
	$(PYTHON) scripts/run_integrated_best.py

verify-mapping-strength:
	$(PYTHON) scripts/verify_mapping_strength_archive.py

check: test verify-mapping-strength
