# Instruções para agentes

Antes de qualquer alteração, leia `.agents/workflow.md`, `.agents/operating-model.md` e o contrato relacionado à tarefa.

## Regras obrigatórias

1. Identifique o integrante pelo pedido e confirme com `git config user.name`/`user.email`. Se houver ambiguidade, pergunte.
2. Nunca mantenha um arquivo compartilhado com o "usuário atual".
3. Verifique alterações existentes e não modifique, descarte ou inclua trabalho alheio.
4. Parta da `main` atualizada e crie uma branch antes de editar: `<tipo>/<integrante>/<descricao-curta>`.
5. Verifique `docs/CHECKLIST.md` em toda tarefa. Atualize-o apenas quando status, escopo, responsável, aceite ou evidência mudar.
6. Implemente e teste em incrementos pequenos. Faça commits semânticos e focados quando a tarefa autorizar commits.
7. Não faça push nem abra PR sem pedido explícito. Nunca faça merge sem confirmação explícita.
8. Nunca faça push direto ou force-push na `main`.
9. No PR, declare se o checklist foi atualizado ou apenas conferido.
10. Documente o sistema que existe; marque planos claramente como pendentes.

## Roteamento

- Dados, EDA e Airflow: `.agents/personas/data-engineer.md`
- Modelo, otimização e observabilidade: `.agents/personas/ml-engineer.md`
- API e nuvem: `.agents/personas/api-cloud-engineer.md`
- CI/CD, qualidade e documentação: `.agents/personas/tech-lead.md`
- Auditoria antes de PR: `.agents/review-checklist.md`

