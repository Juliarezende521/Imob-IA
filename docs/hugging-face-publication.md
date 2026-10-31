# Publicação manual no Hugging Face Hub

## Estado atual

Os pacotes estão somente preparados localmente. Nenhum login, token, repositório
remoto, upload, Inference API ou Space foi usado. A publicação está bloqueada por
`LICENSE_REVIEW_REQUIRED`: o `pyproject.toml` menciona MIT para o software, mas não
há arquivo de licença nem decisão explícita que cubra dataset e modelo.

Antes de publicar, escolha e documente separadamente a licença dos dados e do
modelo. Opções comuns que merecem avaliação são CC BY 4.0 ou CC0 para dados
sintéticos e MIT ou Apache-2.0 para o código/artefato; isto não constitui uma
decisão jurídica. Não substitua `license: other` até a decisão da responsável.

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

## Passos manuais após a aprovação da licença

1. Crie ou revise a conta `<HF_USERNAME>` diretamente no site do Hugging Face.
2. Preencha o perfil com uma bio curta, por exemplo: “AI Engineer em formação,
   construindo sistemas locais e reproduzíveis de NLP em português”.
3. Adicione ao perfil somente links reais e revisados: GitHub e LinkedIn.
4. Crie manualmente um Dataset Repository chamado `<DATASET_REPO_ID>`.
5. Mantenha-o privado durante a revisão, se apropriado.
6. Atualize o YAML do dataset card com a licença efetivamente aprovada.
7. Envie somente os arquivos listados no manifesto do pacote do dataset.
8. Confirme no site as divisões train, validation e test e seus totais.
9. Confirme que nenhum arquivo ou texto do challenge foi enviado.
10. Crie separadamente um Model Repository chamado `<MODEL_REPO_ID>`.
11. Mantenha-o privado durante a revisão, se apropriado.
12. Atualize o YAML do model card com a licença efetivamente aprovada.
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
