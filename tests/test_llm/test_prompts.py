from app.llm.prompts import compliance_prompt


def test_prompt_has_system_and_human_messages():
    messages = compliance_prompt.messages
    roles = [m.__class__.__name__ for m in messages]
    assert "SystemMessagePromptTemplate" in roles
    assert "HumanMessagePromptTemplate" in roles


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
