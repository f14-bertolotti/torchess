
venv/bin/python3:
	python3 -m venv venv
	venv/bin/pip3 install --upgrade pip
	venv/bin/pip3 install -r requirements.txt

install: venv/bin/python3
	python setup.py install

upload:
	venv/bin/python3 setup.py sdist
	venv/bin/python3 -m twine upload dist/*

benches/fig.png: benches/MX150.jsonl benches/RTX40480.jsonl
	jet init --shape 1 1 \
    jet line --input-path benches/MX150.jsonl --x batch_size --y pwn_time --color .7 0 .7 --linestyle "--" --linewidth 2 \
    jet line --input-path benches/MX150.jsonl --x batch_size --y pgx_time --color 1 .7 0 --linestyle "--" --linewidth 2 \
    jet line --input-path benches/RTX40480.jsonl --x batch_size --y pwn_time --color .7 0 .7 --linewidth 2 \
    jet line --input-path benches/RTX40480.jsonl --x batch_size --y pgx_time --color 1 .7 0 --linewidth 2 \
    jet mod --x-label "batch size" --y-label "time (s.)" --x-scale log --y-scale log --right-spine False --top-spine False \
    jet legend --line "torchess MX150" .7 0 .7 2 "--" \
        --line "pgx MX150" 1 .7 0 2 "--" \
        --line "torchess RTX4080" .7 0 .7 2 "-" \
        --line "pgx RTX4080" 1 .7 0 2 "-" \
    jet plot --show False --figsize 10 10 --output-path "benches/fig.png"


clean:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf __pycache__
