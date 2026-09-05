# Guia de uso dos fronts

Este guia permite reproduzir a demonstração dos dois dashboards Streamlit. Use
somente textos clínicos sintéticos: os fronts não são prontuários e não devem
receber dados reais de pacientes.

## Qual front usar

| Front | Público | Objetivo |
|---|---|---|
| `front/app_prod.py` | banca, médico e paciente demonstrativos | apresentar login por papel, RBAC e inferência protegida |
| `front/app_dev.py` | desenvolvedor ou avaliador técnico | inspecionar saúde, modelo, versões, reload e política de idioma |

Os fronts não são redundantes. O portal de produção demonstra a experiência por
papel; o dashboard de desenvolvimento expõe controles técnicos que não devem ser
oferecidos ao paciente.

## Preparação

Suba a API oficial em `http://127.0.0.1:8000` com o modelo e as chaves descritos
no README principal. Em outro terminal PowerShell, configure o portal:

```powershell
$env:TRIAGE_ML_PROD_API_URL = "http://127.0.0.1:8000"
$env:TRIAGE_ML_API_KEY_DOCTOR = "<mesma-chave-da-api>"
$env:TRIAGE_ML_DASHBOARD_DOCTOR_USERNAME = "medico-demo"
$env:TRIAGE_ML_DASHBOARD_DOCTOR_PASSWORD = "<senha-local>"
$env:TRIAGE_ML_DASHBOARD_PATIENT_USERNAME = "paciente-demo"
$env:TRIAGE_ML_DASHBOARD_PATIENT_PASSWORD = "<outra-senha-local>"
uv run streamlit run front/app_prod.py --server.port 8501
```

Em um terceiro terminal, suba o painel técnico:

```powershell
uv run streamlit run front/app_dev.py --server.port 8502
```

- Portal por papel: `http://localhost:8501`
- Dashboard técnico: `http://localhost:8502`

## Casos de uso do portal por papel

### Paciente

1. Entre com a credencial de paciente.
2. Percorra as abas **Entenda o processo**, **Revisão médica** e **Próximos passos**.
3. Confirme que não existe formulário de predição nem resultado do modelo.
4. Mostre que a área informa a necessidade de revisão profissional.
5. Use **Sair** e confirme o retorno ao login.

O que este cenário comprova: negação por padrão, separação de papéis e ausência
de diagnóstico, classe ou score na experiência do paciente.

### Médico

1. Entre com a credencial médica.
2. Confirme que a API está saudável e que o modelo foi carregado.
3. Envie este caso sintético em inglês:

```text
A 62-year-old patient presents with persistent chest pain, shortness of breath,
fatigue, and a history of hypertension. Clinical evaluation is recommended.
```

4. Mostre categoria, versão do modelo, latência e request ID.
5. Destaque o aviso de que a saída apoia a triagem e não constitui diagnóstico.
6. Finalize com **Sair**.

O que este cenário comprova: somente o processo Streamlit autenticado como
médico usa a chave server-side para chamar `POST /predict`.

## Casos de uso do dashboard técnico

1. Em **Health**, confirme `status`, versão e carregamento do modelo.
2. Em **Predição**, reutilize o texto sintético e inspecione a resposta completa.
3. Em **Política de idioma**, execute os presets de texto curto, português e
   inglês válido; confira o status HTTP e o `error_code` esperado.
4. Na sidebar **Modelo**, apresente manifesto, métricas e mapeamento das classes.
5. Em **Trocar modelo**, apenas no ambiente local, liste versões e demonstre o
   reload de um artefato íntegro.

O que este cenário comprova: contrato HTTP, validações, rastreabilidade do
artefato e comportamento técnico reproduzível. Ele não representa uma interface
destinada ao paciente.

## Roteiro curto para o vídeo

1. API saudável e modelo carregado.
2. Login do paciente, proteção clínica e logout.
3. Login do médico, predição sintética e metadados da resposta.
4. Dashboard técnico: health, modelo e política de idioma.
5. Encerrar explicando que RBAC e os dois fronts são extensões demonstrativas do
   projeto, enquanto a inferência, o modelo e o pipeline são os entregáveis centrais.

Nunca mostre senhas, chaves, `.env`, tokens do DagsHub ou dados clínicos reais na
gravação.
