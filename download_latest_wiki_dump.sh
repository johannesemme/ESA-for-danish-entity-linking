#!/bin/sh
LG=$1 # get language
WIKI_FOLDER="wikidata" 
WIKI_DUMP_NAME=${LG}wiki-latest-pages-articles.xml.bz2
WIKI_DUMP_DOWNLOAD_URL=https://dumps.wikimedia.org/${LG}wiki/latest/$WIKI_DUMP_NAME

mkdir -p $WIKI_FOLDER
cd wikidata

if [ ! -f $WIKI_DUMP_NAME ]; # if file does not already exist
then
    echo "Downloading the latest $LG-language Wikipedia dump from $WIKI_DUMP_DOWNLOAD_URL..."
    curl -k -L -s $WIKI_DUMP_DOWNLOAD_URL > $WIKI_DUMP_NAME
    echo "Succesfully downloaded the latest $LG-language Wikipedia dump to $WIKI_DUMP_NAME"
fi