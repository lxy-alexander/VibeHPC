set -e

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.${BUILD_CONTEXT}.txt
# there is a bug in pynvml in current version of container, so we uninstall it
python3 -m pip uninstall -y pynvml
