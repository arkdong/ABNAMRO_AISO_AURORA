# frontend

Streamlit UI for the AURORA profile registry.

## Run

From the repo root:

```bash
uv sync                                   # picks up streamlit
uv run streamlit run frontend/app.py
```

The home page is intentionally empty. Use the sidebar to open **Profiles**,
which has three tabs:

- **View** — every workflow and domain-expert profile, grouped by category
- **Add** — create a new profile (the form fields adapt to the chosen category)
- **Edit** — modify or delete an existing profile

## Persistence

Saves write directly to:

```
profiles/workflow/{id}.yaml
profiles/domain_expert/{id_without_expert_prefix}.yaml
```

After every save / delete the loader's cache is cleared and
`profiles.validate.validate()` is run; any issues are surfaced inline. The
files on disk remain the source of truth — there is no separate database.
