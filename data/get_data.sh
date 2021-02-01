#!/usr/bin/env bash

# taken directly from paper
DATA_URL="https://www.dropbox.com/sh/pma0npf1wr86n9i/AADFc7xWNOe6WilrJQbSHC8Va"

# download data
echo Downloading data!
wget --output-document data.zip $DATA_URL

# unpack it
echo Unpacking data!
unzip data.zip

# flatten hierarchy
echo Cleaning up extra files!
mv CR_10/* .
mv CR_50/* .
rmdir CR_10 CR_50

