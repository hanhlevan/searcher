from app.models.searcher import Searcher

class Main:

    def __init__(self):
        searcher = Searcher("./app/config/service.conf")
        searcher.evaluate()

if __name__ == "__main__":
    Main()