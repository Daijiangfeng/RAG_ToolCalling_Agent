"""Tool package.

Each tool exposes a callable plus a JSON schema so the LLM can select it and
generate arguments (Tool Calling).  The :mod:`registry` aggregates them.
"""
