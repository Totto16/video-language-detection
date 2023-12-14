COPY public.name_basics
from name.basics.tsv with (
        format 'csv',
        delimiter E'\t',
        null '\N',
        header true,
        quote E'\b'
    );
COPY public.title_akas
from title.akas.tsv with (
        format 'csv',
        delimiter E'\t',
        null '\N',
        header true,
        quote E'\b'
    );
COPY public.title_basics
from title.basics.tsv with (
        format 'csv',
        delimiter E'\t',
        null '\N',
        header true,
        quote E'\b'
    );
COPY public.title_crew
from title.crew.tsv with (
        format 'csv',
        delimiter E'\t',
        null '\N',
        header true,
        quote E'\b'
    );
COPY public.title_episode
from title.episode.tsv with (
        format 'csv',
        delimiter E'\t',
        null '\N',
        header true,
        quote E'\b'
    );
COPY public.title_principals
from title.principals.tsv with (
        format 'csv',
        delimiter E'\t',
        null '\N',
        header true,
        quote E'\b'
    );
COPY public.title_ratings
from title.ratings.tsv with (
        format 'csv',
        delimiter E'\t',
        null '\N',
        header true,
        quote E'\b'
    );
