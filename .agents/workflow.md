# Workflow de tarefas

## 1. Entrada e identidade

1. Leia `AGENTS.md` e as instruções relevantes em `.agents/`.
2. Identifique quem solicitou a tarefa pelo texto e por `git config user.name`/`user.email`.
3. Consulte `.agents/team/README.md` para responsabilidade e revisores.
4. Se identidade, objetivo ou critério de aceite forem ambíguos, pare e pergunte.

## 2. Preparação segura

1. Execute `git status` e preserve mudanças existentes.
2. Busque as referências remotas.
3. Garanta que a nova tarefa parte da `main` atualizada.
4. Crie `<tipo>/<integrante>/<descricao-curta>` antes da primeira edição.
5. Para uma tarefa já existente, retome a branch correspondente em vez de abrir outra.

Tipos usuais: `feat`, `fix`, `docs`, `ci`, `test`, `refactor`, `chore`.

## 3. Planejamento e contratos

1. Relacione a tarefa ao `docs/CHECKLIST.md`.
2. Leia `.agents/contracts/README.md` e os contratos afetados.
3. Declare arquivos, testes e evidências esperados.
4. Mudança de arquitetura, escopo ou contrato exige aprovação humana e um ADR quando duradoura.

## 4. Implementação incremental

1. Faça a menor fatia verificável.
2. Execute testes proporcionais ao risco.
3. Registre evidências reproduzíveis.
4. Faça um commit semântico pequeno e prossiga para a próxima fatia.
5. Não misture formatação ampla, refatoração e funcionalidade sem necessidade.

## 5. Checklist e documentação

O checklist é conferido em toda tarefa. Ele só é editado quando houver mudança real em status, escopo, responsável, critério de aceite ou evidência. O README e os documentos devem distinguir claramente `planejado`, `em andamento` e `entregue`.

## 6. Validação e publicação

1. Execute `.agents/review-checklist.md`.
2. Mostre o resumo, os testes e os commits ao solicitante.
3. Push e abertura do PR exigem autorização explícita.
4. Preencha o template do PR, incluindo a situação do checklist.
5. O PR recebe revisão de outro integrante; contratos e arquitetura recebem revisão dos consumidores e de Fábio.
6. Merge na `main` exige confirmação explícita. Force-push na `main` é proibido.

