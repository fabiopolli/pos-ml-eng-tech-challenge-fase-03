# Infraestrutura

Este diretório contém infraestrutura executável local (Compose e observabilidade)
e reserva espaço para infraestrutura como código futura. A decisão de destino em
nuvem está registrada como proposta em
[ADR 0004](../docs/adr/0004-arquitetura-cloud-gcp.md):

- Cloud Run para portal e API de inferência real-time;
- Artifact Registry para imagens por digest;
- Cloud Storage para bundles de modelo imutáveis e somente leitura;
- Cloud Composer + Cloud Run Job para o retreino batch quando houver cadência
  operacional que justifique o custo;
- Secret Manager, service accounts separadas e OIDC/claims para substituir o
  login demonstrativo e as chaves estáticas em acesso humano.

**Estado real:** nenhum recurso GCP, Terraform ou pipeline de deploy cloud foi
provisionado nesta entrega. Docker Compose continua sendo a plataforma
reproduzível local. A implementação cloud só começa após revisão do ADR por
Fábio e dos contratos por Denis e Bill.

