# ADR 0002: Implementação de RBAC Estático na API de Produção

## Contexto
O modelo atua na classificação de categorias clínicas baseada em texto livre. Existe um risco inerente se pacientes tiverem acesso direto às classificações brutas do modelo antes da revisão por um profissional habilitado. A Fase 3 não contempla autenticação complexa de usuários (SSO/JWT).

## Decisão
Implementar um controle de acesso baseado em papéis (RBAC) simplificado utilizando chaves de API estáticas.
1. O acesso será validado pelo header `X-API-Key`.
2. O servidor mapeará chaves distintas injetadas por variáveis de ambiente para os papéis `service`, `doctor` e `patient`.
3. A rota `POST /predict` negará acesso a qualquer chave identificada como `patient`, retornando HTTP 403 (`clinician_review_required`) sem processar o laudo ou vazar métricas textuais nos logs.

O dashboard Streamlit de produção é uma camada de demonstração separada. Ele
autentica localmente médico e paciente por credenciais de ambiente e preserva a
chave de médico apenas no processo Streamlit. A sessão de paciente não chama
`POST /predict`; a proteção efetiva continua sendo aplicada pela API, mesmo que
alguém contorne a interface.

## Consequências
*   **Positivas:** Isola a classificação clínica do usuário final, aplicando *least privilege* em endpoints sensíveis e garantindo a revisão profissional exigida.
*   **Negativas:** Requer o gerenciamento de múltiplas chaves estáticas de acesso (rotação manual), o que atende à Fase 3 mas limita a escalabilidade de identidade para fases futuras.
*   **Limite do dashboard:** login local não equivale a autenticação de produção; deploy em nuvem deve usar um provedor de identidade e um cofre de segredos.
