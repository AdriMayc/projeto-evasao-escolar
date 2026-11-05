"""Streamlit dashboard for the student dropout notebook (same pipeline, interactive)."""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent / "data" / "data.csv"
LABELS = {0: "Não evasão", 1: "Evasão"}

st.set_page_config(page_title="Evasão no Ensino Superior", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = df.columns.str.replace("\t", " ", regex=False).str.strip()
    # Enrolled é removido: trajetória ainda indefinida
    df = df[df["Target"].isin(["Dropout", "Graduate"])].copy()
    df["Alvo"] = df["Target"].map({"Dropout": 1, "Graduate": 0}).astype("int8")
    return df.drop(columns="Target")


def scenarios(df: pd.DataFrame) -> dict[str, list[str]]:
    cols = [c for c in df.columns if c != "Alvo"]
    return {
        "Matrícula": [c for c in cols if "sem" not in c],
        "1º semestre": [c for c in cols if "2nd sem" not in c],
        "2º semestre": cols,
    }


@st.cache_resource
def train_all(df: pd.DataFrame):
    """Train every model x scenario with the notebook's exact settings."""
    models = {
        "Regressão Logística": lambda: LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
        "Árvore de Decisão": lambda: DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, max_depth=8),
        "Random Forest": lambda: RandomForestClassifier(
            class_weight="balanced", n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    }
    out = {}
    for scen, feats in scenarios(df).items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            df[feats], df["Alvo"], test_size=0.30, stratify=df["Alvo"], random_state=RANDOM_STATE)
        scaler = StandardScaler().fit(X_tr)  # fit só no treino, evita data leakage
        for name, make in models.items():
            model = make().fit(scaler.transform(X_tr), y_tr)
            pred = model.predict(scaler.transform(X_te))
            out[(scen, name)] = {
                "model": model, "features": feats, "y_te": y_te, "pred": pred,
                "report": classification_report(y_te, pred, output_dict=True, zero_division=0),
                "matrix": confusion_matrix(y_te, pred),
            }
    return out


def importances(art) -> pd.Series:
    m = art["model"]
    vals = m.feature_importances_ if hasattr(m, "feature_importances_") else abs(m.coef_[0])
    return pd.Series(vals, index=art["features"]).sort_values(ascending=False)


def hbar(s: pd.Series, title: str, color: str = "#4c78a8", diverging: bool = False):
    d = s.rename("value").rename_axis("variable").reset_index()
    enc = {"x": alt.X("value:Q", title=title), "y": alt.Y("variable:N", sort=None, title=None)}
    if diverging:
        enc["color"] = alt.condition(alt.datum.value > 0, alt.value("#d95f59"), alt.value("#4c78a8"))
        return alt.Chart(d).mark_bar().encode(**enc, tooltip=["variable", "value"])
    return alt.Chart(d).mark_bar(color=color).encode(**enc, tooltip=["variable", "value"])


df = load_data()
arts = train_all(df)
scens = scenarios(df)

st.title("Detecção de Evasão no Ensino Superior")
st.caption("Base UCI Dropout and Academic Success · alvo binário: Evasão (1) vs Graduação (0)")

with st.sidebar:
    st.header("Configuração")
    scen = st.selectbox("Cenário de previsão", list(scens), index=1)
    model_name = st.selectbox("Modelo", ["Random Forest", "Árvore de Decisão", "Regressão Logística"])
    st.markdown("---")
    st.caption(f"{len(scens[scen])} variáveis no cenário **{scen}**")

tab_over, tab_corr, tab_model, tab_imp = st.tabs(
    ["Visão geral", "Correlações", "Modelos", "Explicabilidade"])

with tab_over:
    a_min, a_max = int(df["Age at enrollment"].min()), int(df["Age at enrollment"].max())
    age = st.slider("Idade na matrícula", a_min, a_max, (a_min, a_max))
    view = df[df["Age at enrollment"].between(*age)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Estudantes", f"{len(view):,}")
    c2.metric("Evasão", f"{int(view['Alvo'].sum()):,}")
    c3.metric("Taxa de evasão", f"{view['Alvo'].mean():.1%}" if len(view) else "–")

    left, right = st.columns(2)
    with left:
        st.subheader("Distribuição das trajetórias")
        dist = view["Alvo"].map(LABELS).value_counts().rename_axis("Trajetória").reset_index(name="Qtd")
        st.altair_chart(
            alt.Chart(dist).mark_bar().encode(
                x=alt.X("Trajetória:N", sort=None, title=None), y="Qtd:Q",
                color=alt.Color("Trajetória:N", legend=None, scale=alt.Scale(
                    domain=["Evasão", "Não evasão"], range=["#d95f59", "#4c956c"])),
                tooltip=["Trajetória", "Qtd"]),
            width="stretch")
    with right:
        group_options = [c for c in df.columns
                         if c not in ("Alvo", "Age at enrollment") and df[c].nunique() <= 60 and "sem" not in c]
        group = st.selectbox("Taxa de evasão por", group_options,
                             index=group_options.index("Course") if "Course" in group_options else 0)
        st.subheader(f"Taxa de evasão por {group}")
        rate = view.groupby(group)["Alvo"].agg(taxa="mean", n="size").reset_index()
        st.altair_chart(
            alt.Chart(rate).mark_bar(color="#d95f59").encode(
                x=alt.X(f"{group}:N", title=group), y=alt.Y("taxa:Q", axis=alt.Axis(format="%"), title="Taxa"),
                tooltip=[group, alt.Tooltip("taxa:Q", format=".1%"), "n"]),
            width="stretch")

with tab_corr:
    n = st.slider("Quantidade de variáveis (de cada lado)", 3, 15, 5)
    corr = df[scens[scen] + ["Alvo"]].corr(numeric_only=True)["Alvo"].drop("Alvo").sort_values()
    top = pd.concat([corr.head(n), corr.tail(n)])
    st.altair_chart(hbar(top, "Correlação com Alvo", diverging=True), width="stretch")
    st.caption("Correlação é exploratória, não prova causalidade.")

with tab_model:
    art = arts[(scen, model_name)]
    rep = art["report"]["1"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Precision (evasão)", f"{rep['precision']:.3f}")
    c2.metric("Recall (evasão)", f"{rep['recall']:.3f}", help="Métrica principal: falso negativo = aluno em risco sem atenção.")
    c3.metric("F1 (evasão)", f"{rep['f1-score']:.3f}")

    left, right = st.columns(2)
    with left:
        st.subheader(f"Matriz de confusão · {scen}")
        cm = pd.DataFrame(art["matrix"], index=LABELS.values(), columns=LABELS.values())
        cm_long = cm.rename_axis("Real").reset_index().melt("Real", var_name="Previsto", value_name="Qtd")
        base = alt.Chart(cm_long).encode(x="Previsto:N", y="Real:N")
        st.altair_chart(
            base.mark_rect().encode(color=alt.Color("Qtd:Q", scale=alt.Scale(scheme="blues"), legend=None))
            + base.mark_text(fontSize=18).encode(text="Qtd:Q"),
            width="stretch")
    with right:
        st.subheader(f"{model_name} em cada cenário")
        cmp = pd.DataFrame([
            {"Cenário": s, "Métrica": k, "Valor": arts[(s, model_name)]["report"]["1"][v]}
            for s in scens for k, v in {"Precision": "precision", "Recall": "recall", "F1": "f1-score"}.items()])
        st.altair_chart(
            alt.Chart(cmp).mark_bar().encode(
                x=alt.X("Cenário:N", sort=list(scens), title=None), xOffset="Métrica:N",
                y=alt.Y("Valor:Q", scale=alt.Scale(domain=[0, 1])), color="Métrica:N",
                tooltip=["Cenário", "Métrica", alt.Tooltip("Valor:Q", format=".3f")]),
            width="stretch")

    st.subheader("Todos os modelos neste cenário")
    table = pd.DataFrame([
        {"Modelo": m, "Precision": arts[(scen, m)]["report"]["1"]["precision"],
         "Recall": arts[(scen, m)]["report"]["1"]["recall"], "F1": arts[(scen, m)]["report"]["1"]["f1-score"]}
        for m in ["Regressão Logística", "Árvore de Decisão", "Random Forest"]]).sort_values("Recall", ascending=False)
    st.dataframe(table.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
                 hide_index=True, width="stretch")

with tab_imp:
    k = st.slider("Top variáveis", 5, 20, 10)
    imp = importances(arts[(scen, model_name)]).head(k).sort_values()
    label = "Importância média" if model_name != "Regressão Logística" else "|Coeficiente|"
    st.subheader(f"{k} variáveis mais importantes · {model_name} · {scen}")
    st.altair_chart(hbar(imp, label), width="stretch")
    st.caption("Apoia a priorização da coordenação; não substitui a avaliação humana do estudante.")
