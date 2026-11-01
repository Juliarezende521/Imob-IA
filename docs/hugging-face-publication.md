# Publicação manual no Hugging Face Hub

## Estado atual

Os pacotes estão somente preparados localmente. Nenhum login, token, upload,
Inference API ou Space foi usado pelo processo. A proprietária declarou as
licenças em 31 de outubro de 2026:

- código-fonte: MIT;
- dataset sintético: CC BY 4.0 (`cc-by-4.0`);
- modelo TF-IDF + Regressão Logística: Apache 2.0 (`apache-2.0`).

O estado `READY_FOR_REVIEW` significa que a pendência de licença foi resolvida e
que os pacotes passaram pelas validações locais. Ele não significa que houve
upload, publicação, aprovação para produção ou mudança de visibilidade remota.
Os repositórios privados informados pela proprietária são
`JuliaRezende521/imob-ai-intent-dataset-pt` e
`JuliaRezende521/imob-ai-intent-classifier-pt`.

## Preparação e revisão local

```bash
source .venv/bin/activate
python -m pip install -e '.[hub,dev]'
python -m imob_ai.hub.prepare
python -m imob_ai.hub.prepare --validate-only
```

Revise integralmente `build/huggingface/intent-dataset/` e
`build/huggingface/intent-model/`. Confira os hashes dos manifestos, os 384
registros, a ausência do challenge, os tipos confiáveis do `model.skops` e a
decisão de licença. A geração não acessa a rede.

## Passos manuais de revisão e publicação

1. Crie ou revise a conta `<HF_USERNAME>` diretamente no site do Hugging Face.
2. Preencha o perfil com uma bio curta, por exemplo: “AI Engineer em formação,
   construindo sistemas locais e reproduzíveis de NLP em português”.
3. Adicione ao perfil somente links reais e revisados: GitHub e LinkedIn.
4. Crie manualmente um Dataset Repository chamado `<DATASET_REPO_ID>`.
5. Mantenha-o privado durante a revisão, se apropriado.
6. Confirme `license: cc-by-4.0` no YAML do dataset card.
7. Envie somente os arquivos listados no manifesto do pacote do dataset.
8. Confirme no site as divisões train, validation e test e seus totais.
9. Confirme que nenhum arquivo ou texto do challenge foi enviado.
10. Crie separadamente um Model Repository chamado `<MODEL_REPO_ID>`.
11. Mantenha-o privado durante a revisão, se apropriado.
12. Confirme `license: apache-2.0` no YAML do model card.
13. Envie somente os arquivos listados no manifesto do pacote do modelo.
14. Não envie caches, pesos Transformer, tokens, recibos ou dados internos.
15. Substitua referências planejadas apenas por URLs reais de
    `<GITHUB_REPOSITORY_URL>`, `<DATASET_REPO_ID>` e `<MODEL_REPO_ID>`.
16. Teste o exemplo de carregamento em um ambiente local isolado e compatível.
17. Verifique que `model_score` não é apresentado como probabilidade calibrada.
18. Preserve as métricas congeladas e a descrição honesta das limitações.
19. Faça revisão final de privacidade, segurança, licença e arquivos grandes.
20. Só então altere a visibilidade, se a publicação pública tiver sido aprovada.

Não use `push_to_hub` neste fluxo. O envio deliberadamente manual reduz o risco
de publicar arquivos fora dos manifestos. Não crie Space nesta etapa: ainda não
há uma decisão de hospedagem, segurança operacional ou custos, e a demonstração
local existente já cobre a interface.

## Relações entre os artefatos

- GitHub: código-fonte e documentação do processo.
- Dataset Repository: somente as 384 mensagens sintéticas autorizadas.
- Model Repository: baseline clássico reconstruído com train + validation.
- Challenge: privado, excluído de ambos os pacotes e nunca republicado.
- Futura demonstração: deve apontar para artefatos reais somente após revisão.

O conjunto é pequeno e sintético. As métricas não demonstram prontidão para
produção; decisões incertas exigem revisão humana. O baseline permanece a opção
recomendada, enquanto o Transformer é apenas um experimento mais complexo e de
desempenho inferior neste protocolo.

A licença Apache 2.0 do artefato do modelo não substitui nem altera as licenças
de NumPy, scikit-learn, skops ou qualquer outra dependência utilizada.
