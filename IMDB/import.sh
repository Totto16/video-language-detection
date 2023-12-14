#!/usr/bin/env bash

function import() {

    NAME=$1

    mkdir -p downloads
    cd downloads || exit 1

    #  wget "https://datasets.imdbws.com/${NAME}.tsv.gz"

    #  gzip -d "${NAME}.tsv.gz"

    docker exec imdb_database_db mkdir -p /sql/

    docker cp "${NAME}.tsv" "imdb_database_db:/sql/${NAME}.tsv"

    cd ..
}

# import "title.basics"
# import "name.basics"
# import "title.akas"
# import "title.crew"
# import "title.episode"
# import "title.principals"
# import "title.ratings"

docker cp import_data.sql imdb_database_db:/sql/import_data.sql

# docker exec imdb_database_db psql -h localhost -U imdb -d imdb -f imdb_schema.sql

cd downloads || exit 1
docker exec imdb_database_db psql -h localhost -U imdb -d imdb -f /sql/import_data.sql
