
.PHONY: env style quality final clean max.txt

VENV = diachronic-env
export VIRTUAL_ENV := $(abspath ${VENV})
export PATH := ${VIRTUAL_ENV}/bin:${PATH}

${VENV}:
	python3 -m venv $@

env: ${VENV} install_req
	
install_req:	
	pip install --upgrade -r requirements.txt

# black --check --line-length 88 --target-version py38
# isort --check-only ./*.py #Remove these, black and isort contradict each other

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

quality:
	flake8 --extend-ignore E203, W503 --max-line-length 88 src

#Black compatibilty: https://black.readthedocs.io/en/stable/compatible_configs.html
style:
	black --line-length 88 --target-version py38 src
	isort src

final: style clean| ${VENV}
	pip freeze>requirements.txt

