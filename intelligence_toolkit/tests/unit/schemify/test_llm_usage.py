from types import SimpleNamespace

from intelligence_toolkit.schemify.llm import LLMClient
from intelligence_toolkit.schemify.models import SchemifyConfig


def _response(n_searches: int, input_tokens: int = 1_000_000, output_tokens: int = 0):
    output = [SimpleNamespace(type="web_search_call") for _ in range(n_searches)]
    output.append(SimpleNamespace(type="message"))
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                            total_tokens=input_tokens + output_tokens)
    return SimpleNamespace(output=output, usage=usage)


def test_each_web_search_call_is_billed():
    llm = LLMClient(SchemifyConfig(api_key="test", search_model="gpt-5.4-mini",
                                   completion_model="gpt-5.4-mini", cache_enabled=False))
    llm._track_usage(_response(3), is_web_search=True)
    llm._track_usage(_response(0), is_web_search=True)
    llm._track_usage(_response(0), is_web_search=False)
    assert llm.web_search_count == 3
    # 3M input tokens at $0.75/M plus 3 search calls at $0.01 each.
    assert round(llm.total_cost, 4) == round(3 * 0.75 + 3 * 0.01, 4)
