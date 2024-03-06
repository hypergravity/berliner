clean:
	rm -rf build dist berliner.egg-info
	rm -rf .coverage .pytest_cache

install: clean
	pip install .
	rm -rf build dist berliner.egg-info

uninstall:
	pip uninstall berliner -y

test:
	coverage run -m  pytest . --import-mode=importlib --cov-report=html --cov-report=term-missing
	coverage report -m
	rm -rf .coverage .pytest_cache

upload:
	python setup.py sdist bdist_wheel
	twine upload dist/*
