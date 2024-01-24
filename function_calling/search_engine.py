# The following code was adapted from https://github.com/hwchase17/langchain/blob/master/langchain/utilities/google_serper.py

"""Util that calls Google Search using the Serper.dev API."""

import requests
import os

os.environ["SERPER_API_KEY"] = '2b483114459069a7c2c3c98911613f965f54c017'


class GoogleSerperAPIWrapper():
    """Wrapper around the Serper.dev Google Search API.
    You can create a free API key at https://serper.dev.
    To use, you should have the environment variable ``SERPER_API_KEY``
    set with your API key, or pass `serper_api_key` as a named parameter
    to the constructor.
    Example:
        .. code-block:: python
            from langchain import GoogleSerperAPIWrapper
            google_serper = GoogleSerperAPIWrapper()
    """

    def __init__(self, snippet_cnt=10) -> None:
        self.k = snippet_cnt
        self.gl = "cn"
        self.hl = "zh-cn"
        self.serper_api_key = os.environ.get("SERPER_API_KEY", None)
        assert self.serper_api_key is not None, "Please set the SERPER_API_KEY environment variable."
        assert self.serper_api_key != '', "Please set the SERPER_API_KEY environment variable."

    def _google_serper_search_results(self, search_term: str, gl: str, hl: str) -> dict:
        headers = {
            "X-API-KEY": self.serper_api_key or "",
            "Content-Type": "application/json",
        }
        params = {"q": search_term, "gl": gl, "hl": hl}

        url = "https://google.serper.dev/search"
        response = requests.get(url, headers=headers, params=params)
        return response.json()

    def _parse_results(self, results):
        snippets = []

        if results.get("answerBox"):
            answer_box = results.get("answerBox", {})
            if answer_box.get("answer"):
                element = {"content": answer_box.get("answer"), "source": "None"}
                return [element]
            elif answer_box.get("snippet"):
                element = {"content": answer_box.get("snippet").replace("\n", " "), "source": "None"}
                return [element]
            elif answer_box.get("snippetHighlighted"):
                element = {"content": answer_box.get("snippetHighlighted"), "source": "None"}
                return [element]

        if results.get("knowledgeGraph"):
            kg = results.get("knowledgeGraph", {})
            title = kg.get("title")
            entity_type = kg.get("type")
            if entity_type:
                element = {"content": f"{title}: {entity_type}", "source": "None"}
                snippets.append(element)
            description = kg.get("description")
            if description:
                element = {"content": description, "source": "None"}
                snippets.append(element)
            for attribute, value in kg.get("attributes", {}).items():
                element = {"content": f"{attribute}: {value}", "source": "None"}
                snippets.append(element)

        for result in results["organic"][: self.k]:
            if "snippet" in result:
                element = {"content": result["snippet"], "source": result["link"]}
                snippets.append(element)
            for attribute, value in result.get("attributes", {}).items():
                element = {"content": f"{attribute}: {value}", "source": result["link"]}
                snippets.append(element)

        if len(snippets) == 0:
            element = {"content": "No good Google Search Result was found", "source": "None"}
            return [element]

        # keep only the first k snippets
        snippets = snippets[:int(self.k / 2)]

        return snippets

    def run(self, query):
        response = self._google_serper_search_results(query, gl=self.gl, hl=self.hl)
        return self._parse_results(response)


if __name__ == "__main__":
    query = "重庆山王坪门票"
    search = GoogleSerperAPIWrapper()
    print('Google: ', search.run(query))
