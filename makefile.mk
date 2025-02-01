
venv/bin/python3:
	python3 -m venv venv
	venv/bin/pip3 install --upgrade pip
	venv/bin/pip3 install -r requirements.txt

install: venv/bin/python3
	python setup.py install

upload:
	venv/bin/python3 setup.py sdist
	venv/bin/python3 -m twine upload dist/*

clean:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf __pycache__
