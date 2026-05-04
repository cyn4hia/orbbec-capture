# orbbec-capture
auto capture for orbbec

rm -rf .venv

uv venv --python 3.12
source .venv/bin/activate
uv pip install pyorbbecsdk-community

python -c "import pyorbbecsdk; print('ok')"

must be python 3.12