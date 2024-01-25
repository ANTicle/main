import re
def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        # for Azure OpenAI deployment urls
        match = re.search(r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url)
    return match[1]
