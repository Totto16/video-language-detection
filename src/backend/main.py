from typing import Self

from fastapi import FastAPI
from threading import Thread


def register_routes(app) -> None:
    @app.get("/")
    def read_root():
        return {"Hello": "World"}

    @app.get("/items/{item_id}")
    def read_item(item_id: int, q: str | None = None):
        return {"item_id": item_id, "q": q}


def start_app(app) -> None:
    app.start("?")


def stop_app(app) -> None:
    pass


class Backend:
    __handle: Thread
    __app: FastAPI

    def __init__(self: Self) -> None:
        app = FastAPI()
        register_routes(app)
        self.__app = app
        self.__handle = Thread(target=start_app,name="backend-thread" args=(self.__app))

    def start(self: Self) -> None:
        self.__handle.start()

    def stop(self: Self) -> None:
        stop_app(self.__app)
        self.__handle.join()
