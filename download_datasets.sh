wget -P datasets http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip
wget -P datasets http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip
wget -P datasets http://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip

cd datasets

for file in *.zip; do
    unzip "$file" -d .
done