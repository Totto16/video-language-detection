-- CREATE DATABASE imdb;
--
-- taken from https://www.imdb.com/INTerfaces/
--

-- create the table name_basics
CREATE TABLE IF NOT EXISTS public.name_basics (
	nconst TEXT NOT NULL,
	primaryName TEXT NOT NULL,
	birthYear INT NOT NULL,
	deathYear INT,
	primaryProfession TEXT NOT NULL,
	knownForTitles TEXT NOT NULL,
	PRIMARY KEY (nconst)
);
-- create enum for titleType: ATTENTION: New values may be added in the future without warning
CREATE TYPE title_basics_titleType AS ENUM (
	'movie',
	'short',
	'tvseries',
	'tvepisode',
	'video' -- etc. -- TODO: add missing ones
);
-- create enum for genres: ATTENTION: New values may be added in the future without warning
CREATE TYPE title_basics_genres AS ENUM (
	'Documentary',
	'Short',
	'Animation',
	'Comedy',
	'Romance',
	'Sport' -- etc. -- TODO: add missing ones
);
-- create the table title_basics
CREATE TABLE IF NOT EXISTS public.title_basics (
	tconst TEXT NOT NULL,
	titleType title_basics_titleType NOT NULL,
	primaryTitle TEXT NOT NULL,
	originalTitle TEXT NOT NULL,
	isAdult boolean NOT NULL,
	startYear INT NOT NULL,
	endYear INT,
	runTimeMinutes INT NOT NULL,
	genres title_basics_genres [],
	PRIMARY KEY(tconst)
);
-- create enum for type: ATTENTION: New values may be added in the future without warning
CREATE TYPE title_akas_type AS ENUM (
	'alternative',
	'dvd',
	'festival',
	'tv',
	'video',
	'working',
	'original',
	'imdbDisplay'
);
-- create the table title_akas
CREATE TABLE IF NOT EXISTS public.title_akas (
	titleId TEXT NOT NULL,
	ordering INT NOT NULL,
	title TEXT NOT NULL,
	region TEXT NOT NULL,
	language TEXT,
	types title_akas_type [] NOT NULL,
	attributes TEXT [] NOT NULL,
	isOriginalTitle BOOLEAN NOT NULL,
	PRIMARY KEY(titleId, ordering),
	FOREIGN KEY (titleId) REFERENCES public.title_basics(tconst)
);
-- create the table title_crew
CREATE TABLE IF NOT EXISTS public.title_crew (
	tconst TEXT NOT NULL,
	directors TEXT [] NOT NULL,
	writers TEXT [] NOT NULL,
	PRIMARY KEY(tconst),
	-- FOREIGN KEY (EACH ELEMENT OF directors) REFERENCES public.name_basics(nconst),
	-- FOREIGN KEY (EACH ELEMENT OF writers) REFERENCES public.name_basics(nconst),
	FOREIGN KEY (tconst) REFERENCES public.title_basics(tconst)
);
-- create the table title_episode
CREATE TABLE IF NOT EXISTS public.title_episode (
	tconst TEXT NOT NULL,
	parentTconst TEXT NOT NULL,
	seasonNumber INT NOT NULL,
	episodeNumber INT NOT NULL,
	PRIMARY KEY(tconst),
	FOREIGN KEY (parentTconst) REFERENCES public.title_basics(tconst)
);
-- create enum for category: ATTENTION: New values may be added in the future without warning
CREATE TYPE title_principals_category AS ENUM (
	'self',
	'director',
	'cinematographer',
	'composer',
	'editor',
	'actor' -- etc. -- TODO: add missing ones
);
-- create the table title_principals
CREATE TABLE IF NOT EXISTS public.title_principals (
	tconst TEXT NOT NULL,
	ordering INT NOT NULL,
	nconst TEXT NOT NULL,
	category title_principals_category NOT NULL,
	job TEXT,
	characters TEXT [],
	PRIMARY KEY(tconst, ordering),
	FOREIGN KEY (nconst) REFERENCES public.name_basics(nconst),
	FOREIGN KEY (tconst) REFERENCES public.title_basics(tconst)
);
-- create the table title_ratings
CREATE TABLE IF NOT EXISTS public.title_ratings (
	tconst TEXT NOT NULL,
	averageRating NUMERIC NOT NULL,
	numVotes INT NOT NULL,
	PRIMARY KEY(tconst),
	FOREIGN KEY (tconst) REFERENCES public.title_basics(tconst)
);
