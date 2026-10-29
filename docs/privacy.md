# Privacidade e segurança dos dados

## Regra principal

Este projeto utiliza somente dados fictícios, identificados como sintéticos.
Nenhum dado de cliente, imóvel real ou documento interno deve ser copiado para
o repositório, para notebooks ou para registros de execução.

## Dados proibidos

- nomes, documentos, telefones ou e-mails de pessoas reais;
- contratos, mensagens ou endereços pessoais reais;
- credenciais, tokens e chaves de API;
- informações financeiras ou documentos internos.

## Controles iniciais

- arquivos `.env` são ignorados pelo Git;
- `.env.example` contém apenas valores não secretos;
- dados locais e artefatos gerados são ignorados por padrão;
- todo dataset futuro deverá documentar sua geração sintética;
- o diff deverá ser revisado antes de cada commit.
