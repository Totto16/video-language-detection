from dataclasses import dataclass


@dataclass(slots=True, repr=True)
class TraktConfig:
    api_key: str
    cache_path: str


def sync_collection():
    # https://apiz.trakt.tv/sync/collection/media?extended=full,images,available_on&page=1&limit=100&available_on=other&marker=m8hbkkk5nkdlj0hqtgjg10pvvp0

    # sync/collection/

    # "type": media
    # extended=full,images,available_on&
    # available_on=other

    # https://docs.trakt.tv/reference/postsynccollectionadd
    pass


#TODO:
# https://docs.trakt.tv/reference/getsearchlookup
# https://app.trakt.tv/users/me/library?library=other&mode=media
# https://pytrakt.readthedocs.io/en/latest/
