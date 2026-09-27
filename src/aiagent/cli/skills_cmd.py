"""``aiagent skills list`` — show discovered skills (no dspy; stays fast)."""

from __future__ import annotations

import typer

from aiagent.cli._common import (
    CLI_CONTEXT_SETTINGS,
    echo_err,
    examples,
    get_settings,
    print_json,
)
from aiagent.skills.registry import load_registry

_LIST_EXAMPLES = examples(
    ("Built-in and user skills", "aiagent skills list"),
    ("Only your own, from skills_dir", "aiagent skills list --source user"),
    (
        "As JSON, with the reason each broken skill was skipped",
        "aiagent skills list --json",
    ),
)

skills_app = typer.Typer(
    name="skills",
    help="Inspect discovered skills (built-in + user).",
    epilog=_LIST_EXAMPLES,
    no_args_is_help=True,
    add_completion=False,
    context_settings=CLI_CONTEXT_SETTINGS,
)


@skills_app.command("list", epilog=_LIST_EXAMPLES)
def list_skills(
    source: str = typer.Option(
        "all", "--source", help="Which skills: all, builtin or user."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """List skills with their source and description (--json adds the model alias).

    User skills are directories with a SKILL.md under skills_dir (default
    ~/.config/aiagent/skills); one that fails to load is skipped with a warning.
    """
    settings = get_settings()
    registry, errors = load_registry(settings)
    skills = [
        s for s in registry.all() if source == "all" or str(s.source) == source
    ]

    if as_json:
        print_json(
            {
                "skills": [
                    {
                        "name": s.name,
                        "source": str(s.source),
                        "description": s.manifest.description,
                        "model": s.manifest.model,
                    }
                    for s in skills
                ],
                "errors": errors,
            }
        )
        return

    if not skills:
        typer.echo("(no skills found)")
    for s in skills:
        typer.echo(f"{s.name:<14} {str(s.source):<8} {s.manifest.description}")
    for err in errors:
        echo_err(f"warning: skipped skill — {err}")
