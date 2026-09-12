# Architecture Decision Records

Decisões duradouras devem ser registradas como `NNNN-titulo-curto.md`, contendo contexto, opções, decisão, consequências e status.

ADRs inicialmente esperados:

- escolha e recorte do dataset;
- arquitetura de inferência real-time e retreino batch em nuvem;
- formato/otimização do artefato do modelo;
- estratégia de tradução, apenas se necessária.

ADRs registrados:

- [0001 — Escolha e recorte do dataset](0001-escolha-recorte-dataset.md);
- [0002 — RBAC estático na API de produção](0002-rbac-estatico-api-producao.md);
- [0003 — Flexibilizar `sample_size`](0003-flexibilizar-sample-size.md);
- [0004 — Arquitetura GCP para inferência real-time e retreino batch](0004-arquitetura-cloud-gcp.md)
  — proposta aguardando revisão arquitetural.

