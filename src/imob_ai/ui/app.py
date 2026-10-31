"""Aplicação Streamlit do ImobAI, cliente exclusivo da API HTTP."""

from typing import Optional

import streamlit as st
from pydantic import ValidationError

from imob_ai.ui.api_client import ApiClientError, ImobAiApiClient
from imob_ai.ui.config import UiConfig, UiConfigError
from imob_ai.ui.presenters import (
    ACTION_LABELS,
    SCORE_EXPLANATION,
    format_area,
    format_currency,
    format_score,
    human_follow_up,
    property_summary,
    uncertainty_label,
)
from imob_ai.ui.schemas import (
    AssistantResult,
    PropertySearchInput,
    PropertySearchResult,
    QueryInterpretation,
)

SYNTHETIC_NOTICE = (
    "Projeto educacional. Todos os imóveis, mensagens e resultados apresentados "
    "utilizam dados sintéticos."
)


def api_call_error(error: ApiClientError) -> None:
    st.error(error.public_message)
    with st.expander("Detalhes técnicos"):
        st.code(error.technical_detail)


def nullable_choice(label: str, key: str) -> Optional[bool]:
    choice = st.selectbox(label, ["Qualquer", "Sim", "Não"], key=key)
    return {"Sim": True, "Não": False}.get(choice)


def render_property_cards(result: PropertySearchResult) -> None:
    if not result.results:
        st.info("Nenhum imóvel sintético corresponde aos filtros informados.")
        return
    st.caption(
        f"{result.total} resultado(s) · motor {result.search_engine} · dados sintéticos"
    )
    for item in result.results:
        property_ = item.property
        with st.container(border=True):
            st.subheader(property_summary(property_))
            badges = st.columns(3)
            badges[0].badge(property_.property_type.value.title(), color="blue")
            badges[1].badge(property_.purpose.value.title(), color="green")
            badges[2].badge(
                "Disponível" if property_.available else "Indisponível",
                color="green" if property_.available else "gray",
            )
            st.caption(
                f"ID sintético: {property_.id} · {property_.neighborhood}, "
                f"{property_.city}"
            )
            columns = st.columns(4)
            columns[0].metric("Valor", format_currency(property_.price))
            columns[1].metric("Área", format_area(property_.area_m2))
            columns[2].metric("Dormitórios", property_.bedrooms)
            columns[3].metric("Vagas", property_.parking_spaces)
            st.write(property_.description)
            st.write(
                f"**Banheiros:** {property_.bathrooms} · "
                f"**Aceita animais:** {'Sim' if property_.pets_allowed else 'Não'} · "
                f"**Mobiliado:** {'Sim' if property_.furnished else 'Não'}"
            )
            if property_.features:
                st.write("**Características:** " + " · ".join(property_.features))
            st.caption(
                f"Score lexical: {item.lexical_score} · "
                f"Disponível: {'Sim' if property_.available else 'Não'}"
            )


def render_assistant_result(result: AssistantResult, submitted_message: str) -> None:
    classification = result.classification
    st.caption("Mensagem enviada nesta sessão")
    st.write(submitted_message)
    first, second, third = st.columns(3)
    first.metric("Intenção", classification.intent)
    second.metric("model_score", format_score(classification.model_score))
    third.metric("Ação", ACTION_LABELS.get(result.action, result.action))
    st.badge(f"Intenção: {classification.intent}", color="blue")
    st.caption(SCORE_EXPLANATION)
    if classification.requires_human_review:
        st.warning(uncertainty_label(classification))
        if classification.review_reason:
            st.caption(f"Motivo: {classification.review_reason}")
    else:
        st.success(uncertainty_label(classification))
    if result.interpreted_query is not None:
        render_interpretation(result.interpreted_query)
    if result.requires_clarification:
        st.warning(
            "A busca não foi executada porque uma restrição importante precisa "
            "ser esclarecida. Reformule a mensagem e tente novamente."
        )
    st.write(result.message)
    st.write("**Próximo passo:** " + ACTION_LABELS.get(result.action, result.action))
    if human_follow_up(result.action):
        st.info("Próximo passo seguro: continuar com atendimento humano.")
    if result.search is not None:
        render_property_cards(result.search)


def render_interpretation(interpretation: QueryInterpretation) -> None:
    labels = {
        "purpose": "Finalidade",
        "property_type": "Tipo",
        "city": "Cidade",
        "neighborhood": "Bairro",
        "max_price": "Preço máximo",
        "min_bedrooms": "Dormitórios mínimos",
        "min_bathrooms": "Banheiros mínimos",
        "min_parking_spaces": "Vagas mínimas",
        "pets_allowed": "Aceita animais",
        "furnished": "Mobiliado",
    }
    with st.container(border=True):
        st.subheader("Entendi sua busca como")
        if not interpretation.recognized_constraints:
            st.caption("Nenhum filtro estruturado foi reconhecido.")
        for constraint in interpretation.recognized_constraints:
            label = labels.get(constraint.field, constraint.field)
            value = constraint.value
            if isinstance(value, bool):
                value = "Sim" if value else "Não"
            if constraint.field == "max_price" and isinstance(value, (int, float)):
                value = format_currency(float(value))
            correction = (
                f" (interpretação conservadora de “{constraint.corrected_from}”)"
                if constraint.corrected_from
                else ""
            )
            st.markdown(f"- **{label}:** {value}{correction}")
        for constraint in interpretation.unresolved_constraints:
            label = labels.get(constraint.field, constraint.field)
            st.markdown(f"- **{label}:** {constraint.value} — não encontrado")
        if interpretation.remaining_text:
            st.caption(
                "Características usadas no ranking lexical: "
                f"{interpretation.remaining_text}"
            )


def render_connection(config: UiConfig) -> None:
    st.sidebar.subheader("Conexão com a API")
    check_requested = st.sidebar.button("Verificar conexão")
    if "api_health" not in st.session_state or check_requested:
        try:
            with ImobAiApiClient(config) as client:
                live = client.health_live()
                ready = client.health_ready()
            st.session_state.api_health = (live.status, ready.status, None)
        except ApiClientError as exc:
            st.session_state.api_health = ("offline", "not_ready", exc)
    live_status, ready_status, error = st.session_state.api_health
    if live_status == "alive" and ready_status == "ready":
        st.sidebar.success("API conectada e pronta")
    else:
        st.sidebar.error("API indisponível ou não pronta")
        if error:
            with st.sidebar.expander("Detalhes técnicos"):
                st.code(error.technical_detail)
    st.sidebar.caption(config.api_url)


def render_assistant(config: UiConfig) -> None:
    st.header("Assistente")
    st.write("Classifique uma mensagem e veja a rota segura escolhida pela API.")
    with st.expander("Sugestões de mensagens"):
        st.markdown(
            "- Procuro uma casa para alugar em Franca.\n"
            "- Quero agendar uma visita.\n"
            "- Meu boleto venceu, preciso falar com alguém.\n"
            "- O chuveiro do imóvel parou de funcionar."
        )
    with st.form("assistant_form"):
        message = st.text_area("Mensagem em português", max_chars=500)
        submitted = st.form_submit_button("Analisar mensagem", type="primary")
    if not submitted:
        return
    if not message.strip():
        st.warning("Digite uma mensagem antes de continuar.")
        return
    try:
        with st.spinner("Consultando a API local..."):
            with ImobAiApiClient(config) as client:
                result = client.route(message.strip())
        render_assistant_result(result, message.strip())
    except ApiClientError as exc:
        api_call_error(exc)


def clean_text(value: str) -> Optional[str]:
    stripped = value.strip()
    return stripped or None


def render_search(config: UiConfig) -> None:
    st.header("Buscar imóveis")
    st.write("Combine filtros estruturados com o ranking lexical da API.")
    with st.form("property_search_form"):
        text = st.text_input("Texto livre", placeholder="Ex.: casa com quintal")
        row = st.columns(3)
        purpose = row[0].selectbox("Finalidade", ["Qualquer", "aluguel", "venda"])
        property_type = row[1].selectbox(
            "Tipo", ["Qualquer", "casa", "apartamento", "comercial", "terreno"]
        )
        sort = row[2].selectbox(
            "Ordenação",
            ["relevancia", "menor_preco", "maior_preco", "menor_area", "maior_area"],
        )
        location = st.columns(2)
        city = location[0].text_input("Cidade")
        neighborhood = location[1].text_input("Bairro")
        prices = st.columns(2)
        min_price = prices[0].number_input("Preço mínimo", min_value=0.0, value=None)
        max_price = prices[1].number_input("Preço máximo", min_value=0.0, value=None)
        details = st.columns(4)
        min_area = details[0].number_input(
            "Área mínima (m²)", min_value=0.0, value=None
        )
        bedrooms = details[1].number_input(
            "Dormitórios mínimos", min_value=0, value=None, step=1
        )
        bathrooms = details[2].number_input(
            "Banheiros mínimos", min_value=0, value=None, step=1
        )
        parking = details[3].number_input(
            "Vagas mínimas", min_value=0, value=None, step=1
        )
        options = st.columns(3)
        with options[0]:
            pets = nullable_choice("Aceita animais", "search_pets")
        with options[1]:
            furnished = nullable_choice("Mobiliado", "search_furnished")
        limit = options[2].number_input(
            "Máximo de resultados", min_value=1, max_value=50, value=10, step=1
        )
        submitted = st.form_submit_button("Buscar imóveis", type="primary")
    if not submitted:
        return
    try:
        filters = PropertySearchInput(
            text=clean_text(text),
            purpose=None if purpose == "Qualquer" else purpose,
            property_type=None if property_type == "Qualquer" else property_type,
            city=clean_text(city),
            neighborhood=clean_text(neighborhood),
            min_price=min_price,
            max_price=max_price,
            min_area_m2=min_area,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            parking_spaces=parking,
            pets_allowed=pets,
            furnished=furnished,
            sort=sort,
            limit=limit,
        )
        with st.spinner("Buscando no catálogo sintético..."):
            with ImobAiApiClient(config) as client:
                result = client.search(filters)
        render_property_cards(result)
    except ValidationError as exc:
        st.warning("Revise os filtros: os intervalos informados não são válidos.")
        with st.expander("Detalhes técnicos"):
            st.code(str(exc))
    except ApiClientError as exc:
        api_call_error(exc)


def render_explanation() -> None:
    st.header("Entenda o resultado")
    st.subheader("Classificação de intenções")
    st.write(
        "O classificador identifica o objetivo principal de uma mensagem entre oito "
        "classes: busca, visita, dúvida, documentação, financeiro, manutenção, "
        "atendimento humano e fora de escopo."
    )
    st.markdown(
        "- `BUSCA_IMOVEL`: localizar imóveis\n"
        "- `AGENDAR_VISITA`: solicitar visita\n"
        "- `DUVIDA_IMOVEL`: perguntar sobre um imóvel\n"
        "- `DOCUMENTACAO_LOCACAO`: documentação de locação\n"
        "- `FINANCEIRO_LOCACAO`: boletos e questões financeiras\n"
        "- `MANUTENCAO_IMOVEL`: manutenção do imóvel\n"
        "- `ATENDIMENTO_HUMANO`: pedir uma pessoa\n"
        "- `FORA_DE_ESCOPO`: tema não atendido"
    )
    st.info(SCORE_EXPLANATION)
    st.write(
        "Scores baixos ou margens pequenas acionam revisão humana. Essa é uma "
        "política operacional conservadora, calibrada apenas na validação sintética."
    )
    st.subheader("Resultados congelados")
    st.table(
        {
            "Modelo": ["TF-IDF + LogReg", "Transformer experimental"],
            "F1 macro — teste": ["0,786", "0,334"],
            "F1 macro — challenge": ["0,673", "0,385"],
        }
    )
    st.caption(
        "O challenge foi aberto uma única vez, somente após o congelamento dos modelos."
    )
    st.subheader("Recuperação de imóveis")
    st.write(
        "A busca lexical favorece termos explícitos; a semântica compara significados; "
        "a híbrida combina rankings. A lexical permanece padrão por simplicidade, "
        "latência e resultado equivalente ao híbrido no teste congelado. Um modelo "
        "mais complexo não é automaticamente melhor: a escolha depende de evidência."
    )


def render_about(config: UiConfig) -> None:
    st.header("Sobre o projeto")
    st.write(
        "O ImobAI demonstra engenharia de IA reproduzível em um cenário imobiliário "
        "inteiramente sintético, da validação de dados à entrega por API e interface."
    )
    st.code(
        "Streamlit → FastAPI → classificador de intenção\n"
        "                   → busca lexical\n"
        "                   → roteador seguro"
    )
    st.subheader("Tecnologias e decisões")
    st.markdown(
        "- Python, Pydantic, scikit-learn, FastAPI, HTTPX e Streamlit\n"
        "- TF-IDF + Regressão Logística como classificador padrão\n"
        "- busca lexical como recuperador padrão\n"
        "- Transformer preservado somente como experimento\n"
        "- nenhuma LLM ou geração livre de respostas"
    )
    st.subheader("Etapas concluídas")
    st.write(
        "Catálogo sintético, busca estruturada e lexical, avaliação congelada, "
        "buscas semântica e híbrida experimentais, classificação tradicional, "
        "Transformer experimental, API local e esta interface de demonstração."
    )
    st.subheader("Execução local")
    st.code(
        "uvicorn imob_ai.api.app:create_app --factory --host 127.0.0.1 --port 8000\n"
        f"IMOB_AI_API_URL={config.api_url} streamlit run src/imob_ai/ui/app.py"
    )
    st.link_button(
        "Repositório no GitHub", "https://github.com/Juliarezende521/Imob-IA"
    )
    st.warning(
        "Demonstração educacional: não possui autenticação, rate limiting, dados reais "
        "ou garantias para uso em produção."
    )


def main() -> None:
    st.set_page_config(page_title="ImobAI", page_icon="🏠", layout="wide")
    st.markdown(
        "<style>.block-container{max-width:1180px;padding-top:2rem;}"
        "[data-testid='stMetric']{background:#f5f8fb;border-radius:12px;padding:12px;}"
        "</style>",
        unsafe_allow_html=True,
    )
    st.title("ImobAI")
    st.subheader(
        "Assistente imobiliário com classificação de intenções e busca de imóveis"
    )
    st.info(SYNTHETIC_NOTICE)
    try:
        config = UiConfig.from_env()
    except UiConfigError as exc:
        st.error(str(exc))
        st.stop()
    render_connection(config)
    assistant_tab, search_tab, explanation_tab, about_tab = st.tabs(
        ["Assistente", "Buscar imóveis", "Entenda o resultado", "Sobre o projeto"]
    )
    with assistant_tab:
        render_assistant(config)
    with search_tab:
        render_search(config)
    with explanation_tab:
        render_explanation()
    with about_tab:
        render_about(config)


if __name__ == "__main__":
    main()
