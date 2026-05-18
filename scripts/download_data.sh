#!/bin/bash
# Download datasets for MFT-CPEA

DATASET=${1:-houston}
DATA_DIR=${2:-./data/raw}

mkdir -p "$DATA_DIR"

case $DATASET in
    houston)
        echo "Houston dataset needs to be downloaded manually from:"
        echo "https://drive.google.com/drive/folders/1OnLkDpqMtNJy0DRS6YsKKbSqQiiUSgro"
        echo "Place files in $DATA_DIR/Houston11x11/"
        ;;
    trento)
        echo "Trento dataset needs to be downloaded manually from:"
        echo "https://drive.google.com/drive/folders/1HK3eL3loI4Wd-RFr1psLLmVLTVDLctGd"
        echo "Place files in $DATA_DIR/Trento11x11/"
        ;;
    muufl)
        echo "MUUFL dataset needs to be downloaded manually from:"
        echo "https://drive.google.com/drive/folders/1oTUAE3QiVb80sFNi6rvHukFTfZn-lJR_"
        echo "Place files in $DATA_DIR/MUUFL11x11/"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Available: houston, trento, muufl"
        exit 1
        ;;
esac
