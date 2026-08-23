# Papers de referência

Esta pasta guarda artigos científicos que sustentam decisões técnicas do projeto.

## Medical Abstracts TC evaluation paper

**Título**: *Evaluating Unsupervised Text Classification: Zero-shot and Similarity-based Approaches*.

**Autores**: Tim Schopf, Daniel Braun, Florian Matthes.

**Publicação**: NLPIR 2022 (Bangkok, Tailândia), publicado em 2023 pela ACM. DOI [10.1145/3582768.3582795](https://doi.org/10.1145/3582768.3582795).

**Por que está aqui**: o paper é a referência primária do dataset **Medical Abstracts TC Corpus** usado no projeto (o mesmo recorte de 11.550 treino / 2.888 teste / 5 classes que Denis selecionou e que está documentado em [`../dataset.md`](../dataset.md)). Ele compara abordagens não supervisionadas e zero-shot nesse corpus e é citado como contexto narrativo em [`../plans/PLAN-text-classifier.md`](../plans/PLAN-text-classifier.md) e na seção "Modelo (Bill)" do `README.md`.

**Como usamos**:

- Confirmação do schema, contagens por classe e estratégia de split (Tabela 1 do paper).
- Referência de F1 micro no corpus para as abordagens LSA (31,6), SBERT MiniLM (46,5), Lbl2TransformerVec mpnet (56,5) e DeBERTa zero-shot (57,3). Esses números aparecem no plano como **contexto narrativo**, não como meta.
- Hipóteses H1–H4 dos autores sustentam a escolha de um classificador supervisionado linear (TF-IDF + LogisticRegression / LinearSVC) em vez de embeddings no caminho crítico: PLMs maiores não compensam o custo em inferência, e treinar do zero no domínio não é pior.

**Acesso**: use o DOI oficial acima. O PDF foi removido da árvore atual do repositório;
o histórico não foi reescrito nesta alteração.
