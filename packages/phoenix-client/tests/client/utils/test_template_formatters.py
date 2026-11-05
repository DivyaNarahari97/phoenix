from typing import Any

import pytest

from phoenix.client.utils.template_formatters import (
    FStringBaseTemplateFormatter,
    MustacheBaseTemplateFormatter,
    NoOpFormatterBase,
    TemplateFormatter,
    TemplateFormatterError,
)


@pytest.mark.parametrize(
    "formatter_cls, template, variables, expected_output",
    [
        pytest.param(
            MustacheBaseTemplateFormatter,
            "{{ user_id }}",
            {"user_id": r"2025\l6300"},
            r"2025\l6300",
            id="mustache-escaped-sequence",
        ),
        pytest.param(
            MustacheBaseTemplateFormatter,
            "{{ hello }}",
            {"hello": "world"},
            "world",
            id="mustache-whitespace-both-sides",
        ),
        pytest.param(
            MustacheBaseTemplateFormatter,
            "{{hello}}",
            {"hello": "world"},
            "world",
            id="mustache-no-whitespace",
        ),
        pytest.param(
            MustacheBaseTemplateFormatter,
            "{{  hello}}",
            {"hello": "world"},
            "world",
            id="mustache-whitespace-left",
        ),
        pytest.param(
            MustacheBaseTemplateFormatter,
            "{{hello }}",
            {"hello": "world"},
            "world",
            id="mustache-whitespace-right",
        ),
        pytest.param(
            MustacheBaseTemplateFormatter,
            r"\{{ hello }}",
            {"hello": "world"},
            r"\{{ hello }}",
            id="mustache-escaped-sequence-is-not-replaced",
        ),
        pytest.param(
            MustacheBaseTemplateFormatter,
            "{{ hello }}, {{ world }}",
            {"hello": "1", "world": "2"},
            "1, 2",
            id="mustache-multiple-variables",
        ),
        pytest.param(
            MustacheBaseTemplateFormatter,
            "{{ hello }} + {{hello}} = {{ world }}",
            {"hello": "1", "world": "2"},
            "1 + 1 = 2",
            id="mustache-duplicate-variables",
        ),
        pytest.param(
            MustacheBaseTemplateFormatter,
            "{{ hello }}, {{ world }}",
            {"hello": "world", "world": "hello"},
            "world, hello",
            id="mustache-value-is-variable-name",
        ),
        pytest.param(
            FStringBaseTemplateFormatter,
            "{hello}",
            {"hello": "world"},
            "world",
            id="fstring-single-variable",
        ),
        pytest.param(
            FStringBaseTemplateFormatter,
            "{hello}, {world}",
            {"hello": "1", "world": "2"},
            "1, 2",
            id="fstring-multiple-variables",
        ),
        pytest.param(
            FStringBaseTemplateFormatter,
            "{hello} + {hello} = {world}",
            {"hello": "1", "world": "2"},
            "1 + 1 = 2",
            id="fstring-duplicate-variables",
        ),
        pytest.param(
            FStringBaseTemplateFormatter,
            "{hello}, {world}",
            {"hello": "world", "world": "hello"},
            "world, hello",
            id="fstring-value-is-variable-name",
        ),
        pytest.param(
            FStringBaseTemplateFormatter,
            "{{}}",
            {},
            "{}",
            id="fstring-escaped-empty-braces-are-literal",
        ),
        pytest.param(
            FStringBaseTemplateFormatter,
            "Return {{}} when nothing matches",
            {},
            "Return {} when nothing matches",
            id="fstring-escaped-empty-braces-in-prose",
        ),
        pytest.param(
            FStringBaseTemplateFormatter,
            "{hello:>5}",
            {"hello": "x"},
            "    x",
            id="fstring-format-spec-is-preserved",
        ),
        pytest.param(
            NoOpFormatterBase,
            "Hello, world!",
            {},
            "Hello, world!",
            id="noop-formatter-no-change",
        ),
    ],
)
def test_formatters_produce_expected_output(
    formatter_cls: type[TemplateFormatter],
    template: str,
    variables: dict[str, Any],
    expected_output: str,
) -> None:
    formatter = formatter_cls()
    output = formatter.format(template, variables=variables)
    assert output == expected_output


@pytest.mark.parametrize(
    "template",
    [
        pytest.param("{}", id="only-an-empty-field"),
        pytest.param("Return {} when nothing matches", id="empty-field-in-prose"),
        pytest.param("{} {}", id="two-empty-fields"),
        pytest.param("{hello} {}", id="named-field-then-empty-field"),
    ],
)
def test_fstring_automatic_field_numbering_raises_template_formatter_error(
    template: str,
) -> None:
    """Automatic field numbering has no value to draw from, so it must be reported.

    `variables` is a name-to-value mapping, so `str.format` previously surfaced a raw
    `IndexError` ("Replacement index 0 out of range") rather than the documented
    `TemplateFormatterError`.
    """
    formatter = FStringBaseTemplateFormatter()
    with pytest.raises(TemplateFormatterError, match=r"automatic field numbering"):
        formatter.format(template, variables={"hello": "world"})
