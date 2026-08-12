# Workflow agêntico para o time

Este guia permite que qualquer integrante use o Codex com o mesmo processo de colaboração. As regras executáveis estão em `AGENTS.md` e `.agents/`.

## Preparação inicial

1. Instale o Git e o Codex.
2. Clone o repositório e abra sua raiz como projeto no Codex.
3. Configure sua identidade:

```bash
git config user.name "Seu Nome"
git config user.email "seu-email@exemplo.com"
```

4. Diga ao Codex quem você é e qual tarefa assumirá.

O agente cruza essa informação com o Git. Se houver dúvida, ele pergunta; nenhum arquivo compartilhado guarda o usuário atual.

## Fluxo normal

1. O agente lê `AGENTS.md` e identifica a pessoa/responsabilidade.
2. Confere o working tree e preserva trabalho existente.
3. Atualiza referências e parte da `main`.
4. Cria `<tipo>/<integrante>/<descricao-curta>` antes de editar.
5. Confere o checklist e os contratos.
6. Implementa e testa pequenas fatias, criando commits semânticos.
7. Atualiza documentos e evidências quando necessário.
8. Com autorização, faz push e abre um PR.
9. Outro integrante revisa; o merge só ocorre após autorização explícita.

## Quando o checklist muda

`docs/CHECKLIST.md` é conferido em toda tarefa. Ele é editado somente quando mudar status, escopo, responsável, aceite ou evidência. Todo PR marca uma opção:

- checklist atualizado; ou
- checklist conferido, sem alteração necessária.

## O que exige autorização

- Commits locais podem ser criados quando a tarefa autorizar a implementação.
- Push e abertura de PR exigem pedido como “publique”, “suba” ou equivalente.
- Merge na `main` sempre exige confirmação explícita.
- Force-push na `main` é proibido.

## Prompts prontos

### Denis — EDA e dataset

> Sou o Denis. Vou iniciar o EDA dos datasets candidatos. Siga o AGENTS.md, crie a branch apropriada, mantenha os dados fora do Git e produza evidências para decidirmos o dataset. Faça commits pequenos, mas não publique sem me mostrar a validação.

### Bill — baseline do classificador

> Sou o Bill. Implemente comigo o baseline TF-IDF + classificador seguindo o contrato do modelo. Crie a branch da tarefa, fixe seeds, adicione testes e registre métricas reproduzíveis. Não faça push ainda.

### Romário — API

> Sou o Romário. Quero construir o primeiro endpoint `/predict`. Leia o contrato, crie minha branch, implemente uma fatia mínima com testes e documente como executar. Pare antes do push.

### Romário — arquitetura em nuvem

> Sou o Romário. Compare a proposta GCP com os requisitos do projeto e escreva um ADR para inferência real-time e retreino batch. Não alegue recursos implantados. Siga os gates do workflow.

### Fábio — CI/CD

> Sou o Fábio. Amplie o CI para validar o componente recém-integrado. Crie uma branch `ci/fabio/...`, preserve mudanças alheias, execute os testes e prepare um PR com evidências.

### Revisar um PR

> Revise este PR seguindo `.agents/review-checklist.md`. Não altere nem faça merge. Priorize defeitos funcionais, contratos quebrados, dados sensíveis, reprodutibilidade e ausência de evidências.

### Retomar uma tarefa

> Sou [nome] e quero retomar a tarefa [descrição]. Localize e atualize a branch existente; não crie outra se ela já representar o mesmo trabalho. Mostre o estado antes de editar.

### Correção urgente

> Sou [nome]. Há uma correção urgente em [componente]. Siga o fluxo normal com uma branch `fix/[nome]/...`, teste a regressão e peça autorização antes de publicar ou fazer merge.

## Conflitos e trabalho existente

Se o agente encontrar mudanças não relacionadas, ele não as descarta nem as inclui no commit. Deve trabalhar ao redor delas ou parar e explicar o conflito. Antes de integrar, a branch deve incorporar a `main` atual e repetir as validações relevantes.

## Proteção futura da main

Após o primeiro CI verde, o time deve ativar PR obrigatório, checks obrigatórios, bloqueio de force-push e ao menos uma aprovação. Enquanto a proteção técnica não estiver ativa, o workflow continua proibindo push direto na `main`.

