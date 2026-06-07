from app.llm.prompts import compliance_prompt


def test_prompt_has_two_messages():
    assert len(compliance_prompt.messages) == 2


def test_prompt_requires_requirement_and_context():
    assert "requirement" in compliance_prompt.input_variables
    assert "context" in compliance_prompt.input_variables


def test_prompt_renders_with_values():
    rendered = compliance_prompt.format_messages(
        requirement="Risk factors must be disclosed",
        context="[Page 12]\nThe company faces market risks.",
    )
    full_text = " ".join(m.content for m in rendered)
    assert "Risk factors must be disclosed" in full_text
    assert "The company faces market risks" in full_text


def test_system_message_instructs_verbatim_evidence():
    rendered = compliance_prompt.format_messages(
        requirement="test requirement",
        context="test context",
    )
    system_content = rendered[0].content
    assert "verbatim" in system_content.lower()
    assert "json" in system_content.lower()
