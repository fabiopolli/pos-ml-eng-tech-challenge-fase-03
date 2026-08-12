# Como contribuir

Leia `docs/WORKFLOW_AGENTICO.md` e `AGENTS.md` antes de começar.

- Crie uma branch `<tipo>/<integrante>/<descricao-curta>` a partir da `main` atualizada.
- Use commits semânticos pequenos, por exemplo `feat: add prediction schema`.
- Rode `uv run ruff check .` e `uv run pytest`.
- Confira `docs/CHECKLIST.md` e indique no PR se houve atualização.
- Não envie dados, segredos, modelos grandes ou ambientes locais.
- Solicite revisão de outro integrante. Mudanças de contrato, arquitetura, Compose e CI também passam por Fábio.
- Não faça push direto nem force-push na `main`; merge exige confirmação explícita.

