COPY public.title_basics
FROM '/sql/title.basics.tsv' WITH (
        FORMAT 'csv',
        DELIMITER E'\t',
        NULL '\N',
        HEADER true,
        QUOTE E'"'
    );
/* COPY public.name_basics
FROM '/sql/name.basics.tsv' WITH (
        FORMAT 'csv',
        DELIMITER E'\t',
        NULL '\N',
        HEADER true,
        QUOTE E'\b'
    );
COPY public.title_akas
FROM '/sql/title.akas.tsv' WITH (
        FORMAT 'csv',
        DELIMITER E'\t',
        NULL '\N',
        HEADER true,
        QUOTE E'\b'
    );
COPY public.title_crew
FROM '/sql/title.crew.tsv' WITH (
        FORMAT 'csv',
        DELIMITER E'\t',
        NULL '\N',
        HEADER true,
        QUOTE E'\b'
    );
COPY public.title_episode
FROM '/sql/title.episode.tsv' WITH (
        FORMAT 'csv',
        DELIMITER E'\t',
        NULL '\N',
        HEADER true,
        QUOTE E'\b'
    );
COPY public.title_principals
FROM '/sql/title.principals.tsv' WITH (
        FORMAT 'csv',
        DELIMITER E'\t',
        NULL '\N',
        HEADER true,
        QUOTE E'\b'
    );
COPY public.title_ratings
FROM '/sql/title.ratings.tsv' WITH (
        FORMAT 'csv',
        DELIMITER E'\t',
        NULL '\N',
        HEADER true,
        QUOTE E'\b'
    );
 */
