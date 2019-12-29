#!/usr/bin/python3
from app.config.config import configer
from app.models.searcher import Searcher
from flask import Flask
from flask import request, jsonify

class Servicer:

    __delimeter = ","

    def __init__(self):
        self.__searcher = Searcher("./app/config/service.conf")
        # while True:
        #     query = input("Enter your query: ")
        #     searcher.run(query, [""])
    def search(self, query, histories=None):
        query = query.lower()
        if histories is None: histories = [""]
        else: histories = histories.split(self.__delimeter)
        results = self.__searcher.run(query, histories)
        listIDs = results["results"]
        for i, item in enumerate(listIDs):
            for key, val in item.items(): listIDs[i][key] = str(val)
        timeSpent = results["time"]
        dataQuery = results["dataQuery"]
        return {
            "raw_query" : query,
            "listIDs" : listIDs,
            "time" : timeSpent,
            "dataQuery" : dataQuery
        }

servicer = Servicer()
app = Flask(__name__)
@app.route("/", methods=['GET', 'POST'])
def search():
    if request.method == "POST":
        query = request.form.get('query')
        histories = request.form.get("histories")
        data = servicer.search(query, histories)
        print(data)
        return jsonify(isError=False, message="Success", statusCode=200, data=data), 200

if __name__ == '__main__':
    host, port = configer.config["server"], int(configer.config["port"])
    app.run(host=host, port=port)