# Modelo operacional

## Princípios

- Integrar por contratos e fatias verticais, não por silos concluídos isoladamente.
- Preservar reprodutibilidade: parâmetros, seeds, versões, comandos e evidências.
- Manter dados clínicos, segredos e artefatos grandes fora do Git.
- Fazer afirmações verificáveis; planos não contam como entregas.
- Otimização precisa comparar baseline e variante sob as mesmas condições.

## Gates humanos

Exigem decisão humana antes de prosseguir:

- troca do dataset ou alteração relevante de seu schema;
- mudança dos labels ou do contrato de predição;
- escolha final de nuvem e serviços gerenciados;
- inclusão de tradução no caminho online;
- alteração da divisão de responsabilidades;
- publicação externa, merge e mudanças destrutivas.

## Governança Git

- `main` representa estado integrável, ainda que inicialmente sem proteção técnica.
- Toda tarefa normal usa branch e PR.
- Commits seguem Conventional Commits e não incluem alterações alheias.
- Push/PR requer pedido explícito; merge requer confirmação explícita.
- Após o primeiro CI verde, registrar e ativar proteção de branch quando o time estiver pronto.

## Revisão

- Um integrante diferente do autor revisa cada PR.
- Fábio revisa arquitetura, contratos, Compose e CI.
- O responsável pelo componente consumidor revisa mudanças de contrato.
- QA verifica comportamento, evidência e documentação, não apenas estilo.

