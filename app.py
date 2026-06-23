import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import re
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import unicodedata

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Dashboard Legislativo 2023-2026", page_icon="🏛️", layout="wide")
template_grafico = "plotly_white"

# --- 1. FUNÇÕES DE CARREGAMENTO DE DADOS (COM CACHE) ---

STOPWORDS_LEGISLATIVAS = {
    # Verbos de ação genéricos
    'alteração', 'criação', 'instituição', 'estabelecimento', 'regulamentação',
    'dispõe', 'dispoe', 'denominação', 'concessão', 'autorização', 'proibição',
    'inclusão', 'exclusão', 'revogação', 'acréscimo', 'adequação', 'fixação',
    'determinação', 'definição', 'vedação', 'obrigatoriedade', 'implementação',
    # Termos processuais/legislativos genéricos
    'lei federal', 'lei complementar', 'projeto de lei', 'medida provisória',
    'decreto', 'norma', 'dispositivo', 'artigo', 'parágrafo', 'inciso',
    'regulamento', 'normatização', 'legislação', 'lei',
    'código', 'estatuto', 'diretrizes', 'critério', 'critérios', 'prazo',
    'programa', 'política pública', 'política', 'plano', 'programa (administração)',
    # Termos de resultado genéricos
    'incentivo', 'benefício', 'proteção', 'promoção', 'fomento', 'apoio',
    'redução', 'aumento', 'ampliação', 'suspensão', 'utilização', 'combate',
    'fiscalização', 'controle', 'gestão', 'financiamento',
}

def obter_frequencia_palavras_chave(serie_palavras):
    # 1. Transforma em minúsculo e remove o ponto final
    pc = serie_palavras.astype(str).str.lower().str.replace('.', '', regex=False)

    # 2. Separa por vírgula e explode as listas em várias linhas
    pc = pc.str.split(',').explode()

    # 3. Remove espaços em branco
    pc = pc.str.strip()

    # 4. Desconsidera strings vazias ou "nan"
    pc = pc[(pc != '') & (pc != 'nan')]

    # 5. Remove stopwords legislativas genéricas
    pc = pc[~pc.isin(STOPWORDS_LEGISLATIVAS)]

    # 6. Remove palavras muito curtas (1-2 caracteres)
    pc = pc[pc.str.len() > 2]

    # 7. Conta a frequência
    freq = pc.value_counts()

    return freq

def limpar_id(serie):
    """
    Converte uma coluna de ID que pode vir como float (ex: 123.0, por causa
    de NULLs no LEFT JOIN) em string limpa "123", sem o ".0".

    Antes: serie.astype(str).str.split('.').str[0]
    Isso é lento porque o pandas faz, célula por célula em Python puro:
    1) converte pra string, 2) cria uma lista splitando por '.', 3) pega o
    primeiro item da lista. Três passadas, todas em Python.

    Agora: convertemos para Int64 (inteiro anulável do pandas, suporta NaN)
    usando pd.to_numeric, que é vetorizado em C, e só then convertemos pra
    string. NaN/None viram <NA> -> string "<NA>", então tratamos via where.
    """
    numerico = pd.to_numeric(serie, errors='coerce').astype('Int64')
    return numerico.astype(str).where(numerico.notna(), None)


@st.cache_data
def carregar_tudo():
    _t = {}  # dicionário de medições: nome_etapa -> segundos
    _t0 = time.time()

    # Conecta ao banco de dados SQLite
    conn = sqlite3.connect('banco.db')
    cursor = conn.cursor()

    # A. GASTOS + PERFIL (P1, P4, P13)
    query_principal = """
    SELECT 
        d.dep_id AS id_oficial,
        d.dep_nome_eleitoral AS txNomeParlamentar,
        d.dep_escolaridade AS escolaridade,
        desp.desp_sigla_partido AS sgPartido,
        desp.desp_sigla_uf AS sgUF,
        desp.desp_valor_liquido AS vlrLiquido,
        desp.desp_desc_subcota AS txtDescricao
    FROM Despesa desp
    LEFT JOIN Deputado d ON desp.dep_id = d.dep_id
    """
    _s = time.time()
    df_principal = pd.read_sql_query(query_principal, conn)
    _t['A_sql_principal'] = time.time() - _s

    _s = time.time()
    df_principal['id_oficial'] = limpar_id(df_principal['id_oficial'])
    _t['A_pandas_limpar_id'] = time.time() - _s

    # B. FREQUÊNCIA (P11a)
    # Conta presenças apenas em Sessões Deliberativas
    query_presenca = """
    SELECT p.dep_id AS id_oficial, COUNT(p.evt_id) as qtd
    FROM PresencaDeputado p
    JOIN Evento e ON p.evt_id = e.evt_id
    WHERE e.evt_tipo = 'Sessão Deliberativa'
    GROUP BY p.dep_id
    """
    _s = time.time()
    df_pres = pd.read_sql_query(query_presenca, conn)
    _t['B_sql_presenca'] = time.time() - _s

    _s = time.time()
    df_pres['id_oficial'] = limpar_id(df_pres['id_oficial'])
    mapeamento_partido = df_principal[['id_oficial', 'sgPartido']].drop_duplicates()
    df_freq_final = pd.merge(df_pres, mapeamento_partido, on='id_oficial')
    frequencia_partido = df_freq_final.groupby('sgPartido')['qtd'].mean()
    _t['B_pandas_merge_groupby'] = time.time() - _s

    # C. PRODUÇÃO (P11b)
    query_autores = """
    SELECT 
        dep_id AS id_oficial, 
        prop_id AS idProposicao_link 
    FROM ProposicaoAutor
    WHERE dep_id IS NOT NULL
    """
    _s = time.time()
    df_autores = pd.read_sql_query(query_autores, conn)
    _t['C_sql_autores'] = time.time() - _s

    _s = time.time()
    df_autores['id_oficial'] = limpar_id(df_autores['id_oficial'])
    df_autores['idProposicao_link'] = limpar_id(df_autores['idProposicao_link'])
    _t['C_pandas_limpar_id'] = time.time() - _s

    # D. EMENTAS PARA NUVEM E TEMAS
    query_prop = "SELECT prop_id AS idProposicao_link, prop_ementa AS ementa, prop_palavras_chave AS palavras_chave, prop_sigla_tipo || ' ' || prop_numero || '/' || prop_ano AS nome_projeto FROM Proposicao"
    _s = time.time()
    df_prop = pd.read_sql_query(query_prop, conn)
    _t['D_sql_proposicao'] = time.time() - _s

    _s = time.time()
    df_prop['idProposicao_link'] = limpar_id(df_prop['idProposicao_link'])
    _t['D_pandas_limpar'] = time.time() - _s

    # E. CLASSIFICAÇÃO TEMÁTICA (P3)
    # IMPORTANTE: classificamos por proposição ÚNICA (df_prop), não pelo
    # merge com autores (df_temas antigo). Uma proposição com 3 autores
    # antes tinha sua ementa reprocessada 3x pelo str.count — desperdício
    # puro, já que o tema de uma proposição não depende de quem a assina.
    TEMAS_KEYWORDS = {
        'Saúde':          ['saúde', 'médic', 'hospital', 'sus', 'doença', 'farmac', 'vacin', 'enferm', 'sanitár', 'anvisa', 'medicamento'],
        'Educação':       ['educaç', 'escola', 'ensino', 'universidade', 'professor', 'aluno', 'magistério', 'bolsa', 'enem', 'fundeb', 'creche'],
        'Segurança':      ['segurança', 'polici', 'crime', 'pena', 'presídio', 'violência', 'penal', 'armamento', 'arma', 'tráfico'],
        'Tributário':     ['tribut', 'imposto', 'taxa', 'fiscal', 'icms', 'iss', 'irpf', 'receita federal', 'alíquota', 'contribuição'],
        'Econômico':      ['econom', 'indústria', 'comercio', 'empresa', 'empreend', 'crédito', 'banco', 'financ', 'mercado', 'investimento'],
        'Ambiental':      ['ambient', 'floresta', 'desmatamento', 'clima', 'carbono', 'sustentáv', 'poluiç', 'resíduo', 'biodiversidade'],
        'Infraestrutura': ['infraestrutura', 'rodovia', 'ferrovia', 'porto', 'aeroporto', 'saneamento', 'habitação', 'obras', 'energia elétrica'],
        'Social':         ['social', 'família', 'criança', 'idoso', 'mulher', 'pessoa com deficiência', 'vulneráv', 'pobreza', 'bolsa família'],
        'Agropecuário':   ['agrícol', 'agropec', 'rural', 'agricultor', 'pecuári', 'soja', 'agroneg'],
    }

    _s = time.time()
    if not df_prop.empty:
        textos = df_prop['ementa'].fillna('').str.lower()
        # Compila cada regex uma única vez, fora de qualquer loop por linha.
        patterns_compilados = {
            tema: re.compile('|'.join(palavras))
            for tema, palavras in TEMAS_KEYWORDS.items()
        }
        scores_df = pd.DataFrame(index=df_prop.index)
        for tema, pattern in patterns_compilados.items():
            # Passar o Pattern já compilado evita recompilar a regex
            # a cada chamada interna do str.count.
            scores_df[tema] = textos.str.count(pattern)
        max_scores = scores_df.max(axis=1)
        df_prop_classificado = df_prop.copy()
        df_prop_classificado['tema'] = scores_df.idxmax(axis=1).where(max_scores > 0, other=None)
        df_prop_classificado = df_prop_classificado.dropna(subset=['tema'])
    else:
        df_prop_classificado = pd.DataFrame(columns=['idProposicao_link', 'ementa', 'palavras_chave', 'nome_projeto', 'tema'])
    _t['E_pandas_classificacao_tematica'] = time.time() - _s

    # Agora sim, mergeia com autores (aqui pode duplicar por autor — é o
    # esperado, pois queremos saber quais deputados assinaram cada tema)
    _s = time.time()
    df_temas = pd.merge(df_autores, df_prop, on='idProposicao_link', how='inner')
    df_temas_classificado = pd.merge(
        df_autores,
        df_prop_classificado,
        on='idProposicao_link',
        how='inner'
    )
    _t['E_pandas_merge_autores'] = time.time() - _s

    # F. VOTOS POR TEMA (P3)
    query_votos_p3 = """
    SELECT 
        vd.dep_id AS id_oficial,
        vd.voto_opcao AS voto,
        v.prop_id AS idProposicao_link,
        v.vot_id AS id_votacao_str,
        v.vot_registro AS hora_votacao,
        v.vot_descricao AS descricao_votacao,
        v.vot_aprovada AS resultado_aprovado
    FROM VotoDeputado vd
    JOIN Votacao v ON vd.vot_id = v.vot_id
    WHERE v.prop_id IS NOT NULL
    """
    _s = time.time()
    df_votos_detalhado = pd.read_sql_query(query_votos_p3, conn)
    _t['F_sql_votos_p3'] = time.time() - _s

    _s = time.time()
    df_votos_detalhado['id_oficial'] = limpar_id(df_votos_detalhado['id_oficial'])
    df_votos_detalhado['idProposicao_link'] = limpar_id(df_votos_detalhado['idProposicao_link'])
    df_votos_detalhado['id_votacao_str'] = df_votos_detalhado['id_votacao_str'].astype(str).str.strip()
    df_votos_temas = pd.merge(df_votos_detalhado, df_temas_classificado[['idProposicao_link', 'tema', 'ementa', 'nome_projeto']], on='idProposicao_link', how='inner')
    _t['F_pandas_limpar_merge'] = time.time() - _s

    conn.close()

    _t['TOTAL'] = time.time() - _t0

    # --- Relatório de tempos: imprime no terminal onde o streamlit roda ---
    print("\n" + "=" * 60)
    print("DIAGNÓSTICO DE TEMPO — carregar_tudo()")
    print("=" * 60)
    sql_total = sum(v for k, v in _t.items() if '_sql_' in k)
    pandas_total = sum(v for k, v in _t.items() if '_pandas_' in k)
    for nome, segundos in _t.items():
        print(f"{nome:35s} {segundos:8.3f}s")
    print("-" * 60)
    print(f"{'TOTAL SQL (read_sql_query)':35s} {sql_total:8.3f}s")
    print(f"{'TOTAL PANDAS (processamento)':35s} {pandas_total:8.3f}s")
    print("=" * 60 + "\n")

    return df_principal, frequencia_partido, df_autores, df_temas, df_temas_classificado, df_votos_temas, TEMAS_KEYWORDS


@st.cache_data
def carregar_votos_deputado(dep_id_str):
    conn = sqlite3.connect('banco.db')
    query = """
    SELECT 
        vd.dep_id AS id_oficial,
        vd.voto_opcao AS voto,
        v.prop_id AS idProposicao_link,
        v.vot_id AS id_votacao_str,
        v.vot_registro AS hora_votacao,
        v.vot_descricao AS descricao_votacao,
        v.vot_aprovada AS resultado_aprovado
    FROM VotoDeputado vd
    JOIN Votacao v ON vd.vot_id = v.vot_id
    WHERE v.prop_id IS NOT NULL AND vd.dep_id = ?
    """
    df = pd.read_sql_query(query, conn, params=(dep_id_str,))
    conn.close()
    if df.empty:
        return pd.DataFrame()
    df['id_oficial'] = df['id_oficial'].astype(str).str.split('.').str[0]
    df['idProposicao_link'] = df['idProposicao_link'].astype(str).str.split('.').str[0]
    df['id_votacao_str'] = df['id_votacao_str'].astype(str).str.strip()
    return df  # <-- sem merge aqui

@st.cache_data
def carregar_votos_e_calcular_alinhamento(df_principal_completo):
    conn = sqlite3.connect('banco.db')
    
    vazio = (pd.DataFrame(columns=['sgPartido', 'perc_alinhamento']), pd.DataFrame(), pd.DataFrame(), '', '')

    # E. VOTOS (P10)
    query_votos = """
    SELECT 
        vot_id AS idVotacao, 
        dep_id AS id_oficial, 
        voto_opcao AS voto 
    FROM VotoDeputado
    """
    df_votos = pd.read_sql_query(query_votos, conn)
    
    if df_votos.empty:
        conn.close()
        return vazio

    df_votos['id_oficial'] = df_votos['id_oficial'].astype(str).str.split('.').str[0]

    # Mapeamento Deputado -> Partido
    mapeamento_partido = df_principal_completo[['id_oficial', 'sgPartido']].drop_duplicates()
    df_votos_partido = pd.merge(df_votos, mapeamento_partido, on='id_oficial', how='inner')

    # Filtrar votos válidos (Sim ou Não)
    df_votos_validos = df_votos_partido[df_votos_partido['voto'].isin(['Sim', 'Não'])].copy()

    if df_votos_validos.empty:
        conn.close()
        return vazio

    id_votacao_col = 'idVotacao'
    voto_col = 'voto'

    # Calcular alinhamento geral por partido
    bancada_voto_majoritario = df_votos_validos.groupby([id_votacao_col, 'sgPartido'])[voto_col].agg(
        lambda x: x.value_counts().index[0]
    ).reset_index()
    bancada_voto_majoritario.rename(columns={voto_col: 'voto_bancada'}, inplace=True)

    df_comparacao = pd.merge(df_votos_validos, bancada_voto_majoritario, on=[id_votacao_col, 'sgPartido'])
    df_comparacao['alinhado'] = (df_comparacao[voto_col] == df_comparacao['voto_bancada']).astype(int)

    alinhamento_final = df_comparacao.groupby('sgPartido')['alinhado'].mean() * 100
    df_alinhamento = alinhamento_final.reset_index(name='perc_alinhamento')

    # Carregar metadados das votações
    query_meta = "SELECT vot_id AS id_votacao_str, vot_descricao AS tema_label FROM Votacao"
    df_meta = pd.read_sql_query(query_meta, conn)
    df_meta['id_votacao_str'] = df_meta['id_votacao_str'].astype(str)
    df_meta['tema_label'] = df_meta['tema_label'].fillna(df_meta['id_votacao_str']).astype(str)

    # Calcular % Sim e % Não por partido
    df_votos_validos['id_votacao_str'] = df_votos_validos[id_votacao_col].astype(str).str.strip()
    contagem = df_votos_validos.groupby(['id_votacao_str', 'sgPartido', voto_col]).size().reset_index(name='qtd')
    total_por_vot_partido = contagem.groupby(['id_votacao_str', 'sgPartido'])['qtd'].sum().reset_index(name='total')
    contagem = pd.merge(contagem, total_por_vot_partido, on=['id_votacao_str', 'sgPartido'])
    contagem['perc'] = contagem['qtd'] / contagem['total'] * 100
    contagem.rename(columns={voto_col: 'voto'}, inplace=True)
    contagem = pd.merge(contagem, df_meta, on='id_votacao_str', how='left')
    contagem['tema_label'] = contagem['tema_label'].fillna(contagem['id_votacao_str'])

    # Calcular votações mais divididas
    coesao = df_votos_validos.groupby(['id_votacao_str', 'sgPartido'])[voto_col].apply(
        lambda x: x.value_counts(normalize=True).iloc[0] * 100
    ).reset_index(name='coesao')
    divisao_por_votacao = coesao.groupby('id_votacao_str')['coesao'].mean().reset_index(name='coesao_media')
    divisao_por_votacao = pd.merge(divisao_por_votacao, df_meta, on='id_votacao_str', how='left')
    divisao_por_votacao['tema_label'] = divisao_por_votacao['tema_label'].fillna(divisao_por_votacao['id_votacao_str'])
    divisao_por_votacao.sort_values('coesao_media', inplace=True)

    conn.close()
    return df_alinhamento, contagem, divisao_por_votacao, id_votacao_col, voto_col


@st.cache_data
def calcular_fidelidade_individual():
    conn = sqlite3.connect('banco.db')
    q = """
    SELECT vot_id AS idVotacao, dep_id AS id_oficial, voto_opcao AS voto
    FROM VotoDeputado
    WHERE voto_opcao IN ('Sim', 'Não')
    """
    df_v = pd.read_sql_query(q, conn)
    conn.close()
    df_v['id_oficial'] = df_v['id_oficial'].astype(str).str.split('.').str[0]

    # Voto majoritário da bancada por votação+partido — feito aqui dentro do cache
    # (precisa do mapeamento dep→partido, que buscamos direto do banco)
    conn2 = sqlite3.connect('banco.db')
    mapa = pd.read_sql_query(
        "SELECT DISTINCT d.dep_id AS id_oficial, desp.desp_sigla_partido AS sgPartido "
        "FROM Deputado d JOIN Despesa desp ON d.dep_id = desp.dep_id",
        conn2
    )
    conn2.close()
    mapa['id_oficial'] = mapa['id_oficial'].astype(str).str.split('.').str[0]

    df_vp = pd.merge(df_v, mapa, on='id_oficial', how='inner')

    voto_maj = (
        df_vp.groupby(['idVotacao', 'sgPartido'])['voto']
        .agg(lambda x: x.value_counts().index[0])
        .reset_index(name='voto_bancada')
    )
    df_vp = pd.merge(df_vp, voto_maj, on=['idVotacao', 'sgPartido'])
    df_vp['alinhado'] = (df_vp['voto'] == df_vp['voto_bancada']).astype(int)

    fidelidade_dep = (
        df_vp.groupby('id_oficial')['alinhado']
        .mean().mul(100).round(1)
        .reset_index(name='fidelidade')
    )
    return fidelidade_dep  # <-- já entrega o resultado final


@st.cache_data
def calcular_influencia_p8():
    conn = sqlite3.connect('banco.db')

    # Busca todas as proposições com status de aprovação via votação
    # Uma proposição é "aprovada" se tiver ao menos uma votação com vot_aprovada = 1
    query_p8 = """
    SELECT
        pa.dep_id       AS id_oficial,
        pa.prop_id      AS prop_id,
        MAX(CASE WHEN v.vot_aprovada = 1 THEN 1 ELSE 0 END) AS foi_aprovada
    FROM ProposicaoAutor pa
    LEFT JOIN Votacao v ON pa.prop_id = v.prop_id
    WHERE pa.dep_id IS NOT NULL
    GROUP BY pa.dep_id, pa.prop_id
    """
    df_p8 = pd.read_sql_query(query_p8, conn)
    conn.close()

    df_p8['id_oficial'] = df_p8['id_oficial'].astype(str).str.split('.').str[0]
    df_p8['prop_id']    = df_p8['prop_id'].astype(str).str.split('.').str[0]

    # Agrega por deputado
    resumo = df_p8.groupby('id_oficial').agg(
        total_proposicoes  = ('prop_id', 'count'),
        proposicoes_aprovadas = ('foi_aprovada', 'sum')
    ).reset_index()

    resumo['taxa_aprovacao'] = (
        resumo['proposicoes_aprovadas'] / resumo['total_proposicoes'] * 100
    ).round(1)

    # % em relação ao total de aprovações no período (influência relativa)
    total_aprov_periodo = resumo['proposicoes_aprovadas'].sum()
    resumo['perc_do_total_aprov'] = (
        resumo['proposicoes_aprovadas'] / total_aprov_periodo * 100
        if total_aprov_periodo > 0 else 0
    ).round(2)

    return resumo


def normalizar_fornecedor(s):
    s = s.strip().upper()
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')  # remove acentos
    s = re.sub(r'[.\-/]$', '', s)   # remove ponto/traço/barra no final
    s = re.sub(r'\s+', ' ', s)      # colapsa espaços duplos
    return s.title()


@st.cache_data
def carregar_fornecedores():
    conn = sqlite3.connect('banco.db')
    query_forn = """
    SELECT
        d.dep_id        AS id_oficial,
        d.dep_nome_eleitoral AS txNomeParlamentar,
        desp.desp_sigla_partido  AS sgPartido,
        desp.desp_sigla_uf       AS sgUF,
        desp.desp_fornecedor_nome AS fornecedor,
        desp.desp_fornecedor_doc  AS doc_fornecedor,
        desp.desp_desc_subcota    AS categoria,
        desp.desp_valor_liquido   AS vlrLiquido
    FROM Despesa desp
    LEFT JOIN Deputado d ON desp.dep_id = d.dep_id
    WHERE desp.desp_fornecedor_nome IS NOT NULL
      AND desp.desp_fornecedor_nome != ''
      AND desp.desp_valor_liquido > 0
    """
    df_forn = pd.read_sql_query(query_forn, conn)
    conn.close()
    df_forn['id_oficial'] = df_forn['id_oficial'].astype(str).str.split('.').str[0]
    df_forn['fornecedor'] = df_forn['fornecedor'].dropna().map(normalizar_fornecedor)

    return df_forn


# --- 2. EXECUÇÃO DO CARREGAMENTO ---
with st.spinner('Consultando os dados da 57ª Legislatura...'):
    df_principal, df_freq, df_autores, df_temas, df_temas_classificado, df_votos_temas, TEMAS_KEYWORDS = carregar_tudo()
    df_alinhamento, df_votos_contagem, df_divisao, _id_vot_col, _voto_col = carregar_votos_e_calcular_alinhamento(df_principal)

# --- 3. FILTROS LATERAIS (GLOBAIS) ---
st.sidebar.header("🔍 Filtros Globais")

# 1. Filtro de Nome
# Pega a última combinação de Nome, Partido e Estado de cada deputado (útil caso alguém tenha mudado de partido)
df_nomes = df_principal[['txNomeParlamentar', 'sgPartido', 'sgUF']].dropna().drop_duplicates(subset=['txNomeParlamentar'], keep='last')

# Cria um dicionário que mapeia o "Nome" para "Nome (Partido - Estado)"
mapa_nomes = {"Todos": "Todos"}
for _, row in df_nomes.iterrows():
    nome = row['txNomeParlamentar']
    partido = row['sgPartido']
    uf = row['sgUF']
    mapa_nomes[nome] = f"{nome} ({partido} - {uf})"

# Lista de opções contendo apenas os nomes puros (para o filtro continuar funcionando)
nomes_disp = ["Todos"] + sorted(df_nomes['txNomeParlamentar'].tolist())

# Selectbox usando o format_func para alterar apenas a exibição visual
nome_sel = st.sidebar.selectbox(
    "Buscar por Nome do Deputado", 
    options=nomes_disp,
    format_func=lambda x: mapa_nomes.get(x, x) # <-- Aqui acontece a mágica
)

# 2. Lógica para exibir a foto do deputado selecionado
if nome_sel != "Todos":
    dep_selecionado_row = df_principal[df_principal['txNomeParlamentar'] == nome_sel].iloc[0]
    id_dep = dep_selecionado_row['id_oficial']
    
    # URL da Câmara
    url_foto = f"https://www.camara.leg.br/internet/deputado/bandep/{id_dep}.jpg"
    
    # Exibe a foto. Se o link não carregar, o Streamlit apenas deixa o espaço vazio 
    # ou exibe um pequeno ícone padrão do navegador, sem quebrar o resto do layout.
    st.sidebar.image(url_foto, width=160, caption=nome_sel)
else:
    # Se "Todos" estiver selecionado, simplesmente não exibe nada ou exibe um título
    st.sidebar.markdown("---")
    st.sidebar.write("Selecione um deputado para ver sua foto.")

# 3. Restante dos filtros
partidos_disp = sorted(df_principal['sgPartido'].dropna().unique())
partido_sel = st.sidebar.multiselect("Partidos", partidos_disp)
ufs_disp = sorted(df_principal['sgUF'].dropna().unique())
uf_sel = st.sidebar.multiselect("Estados (UF)", ufs_disp)

df_filtrado = df_principal.copy()
if nome_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado['txNomeParlamentar'] == nome_sel]
if partido_sel:
    df_filtrado = df_filtrado[df_filtrado['sgPartido'].isin(partido_sel)]
if uf_sel:
    df_filtrado = df_filtrado[df_filtrado['sgUF'].isin(uf_sel)]

df_link_partido_filtrado = df_filtrado[['id_oficial', 'sgPartido']].drop_duplicates()

# --- 4. MÉTRICAS GLOBAIS ---
st.title("🏛️ Dashboard Legislativo - 57ª Legislatura (2023-2026)")

# Calcula os valores
total_gasto = df_filtrado['vlrLiquido'].sum()
total_notas = len(df_filtrado)

# Formata pro padrão americano primeiro, depois inverte ponto e vírgula
gasto_br = f"{total_gasto:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
notas_br = f"{total_notas:,}".replace(",", ".")

m1, m2, m3 = st.columns(3)
m1.metric("Gasto Total Acumulado (Cota)", f"R$ {gasto_br}")
m2.metric("Deputados Analisados (no filtro)", df_filtrado['txNomeParlamentar'].nunique())
m3.metric("Notas Fiscais Processadas", notas_br)
st.divider()

def renderizar_p1():
    # --- 5. P1 (RANKING DE GASTOS POR DEPUTADO) ---
    st.header("💰 P1 — Ranking de Gastos por Deputado")

    col_slider, _ = st.columns([0.4, 0.6])
    with col_slider:
        n_top = st.slider("Mostrar quantos deputados no ranking de gastos?", 5, 50, 15)

    # 1. Agrupa e soma os gastos
    ranking_gastos = df_filtrado.groupby(['txNomeParlamentar', 'sgPartido', 'sgUF'])['vlrLiquido'].sum().sort_values(ascending=False).reset_index()

    # 2. Cria a nova coluna com o formato desejado: Nome (Partido - UF)
    ranking_gastos['NomeExibicao'] = ranking_gastos['txNomeParlamentar'] + " (" + ranking_gastos['sgPartido'] + " - " + ranking_gastos['sgUF'] + ")"

    # 3. Atualiza o gráfico para usar a nova coluna no eixo Y
    fig_p1 = px.bar(
        ranking_gastos.head(n_top), 
        x='vlrLiquido', 
        y='NomeExibicao', # <-- Nova coluna sendo usada aqui
        orientation='h', 
        color='vlrLiquido', 
        color_continuous_scale='Reds',
        labels={'vlrLiquido': 'Total Gasto (R$)', 'NomeExibicao': 'Deputado'}, # <-- Label atualizado
        template=template_grafico, 
        # Mantém os dados originais no hover (balão flutuante) para ficar mais limpo
        hover_data={'txNomeParlamentar': True, 'sgPartido': True, 'sgUF': True, 'NomeExibicao': False} 
    )
    
    fig_p1.update_layout(yaxis={'categoryorder':'total ascending'}, height=400 + (n_top * 10))
    st.plotly_chart(fig_p1, width='stretch')
    st.divider()

def renderizar_p2():
    # --- 5b. P2 (AGRUPAMENTO POR EIXO TEMÁTICO) ---
    st.header("🗂️ P2 — Agrupamento por Eixo Temático de Atuação")

    if df_temas_classificado.empty:
        st.warning("⚠️ Nenhuma proposição classificada encontrada.")
    else:
        ids_filtrados = df_filtrado['id_oficial'].unique()
        df_tc_filtrado = df_temas_classificado[df_temas_classificado['id_oficial'].isin(ids_filtrados)]

        if df_tc_filtrado.empty:
            st.info("Nenhuma proposição classificada encontrada para os deputados no filtro atual.")
        else:
            perfil_dep = df_tc_filtrado.groupby(['id_oficial', 'tema']).size().reset_index(name='qtd')
            idx_max = perfil_dep.groupby('id_oficial')['qtd'].idxmax()
            tema_dominante = perfil_dep.loc[idx_max][['id_oficial', 'tema']].rename(columns={'tema': 'tema_dominante'})

            contagem_temas = tema_dominante['tema_dominante'].value_counts().reset_index()
            contagem_temas.columns = ['Tema', 'Deputados']

            prop_por_tema = (
                df_tc_filtrado
                .drop_duplicates(subset=['idProposicao_link'])['tema']
                .value_counts()
                .reset_index()
            )
            prop_por_tema.columns = ['Tema', 'Proposições']

            CORES_TEMAS = {
                'Saúde': '#e74c3c', 'Educação': '#3498db', 'Segurança': '#2c3e50',
                'Tributário': '#f39c12', 'Econômico': '#27ae60', 'Ambiental': '#1abc9c',
                'Infraestrutura': '#8e44ad', 'Social': '#e91e63', 'Agropecuário': '#795548',
            }

            tab_p2a, tab_p2b, tab_p2c, tab_p2d = st.tabs([
                "📊 Proposições por Tema", "👥 Deputados por Tema", "🔍 Perfil Individual", "☁️ Nuvem por Tema"
            ])

            with tab_p2a:
                # 1. Ordenar os dados e mapear as cores exatas para esta ordem
                df_plot_a = prop_por_tema.sort_values('Proposições')
                cores_a = df_plot_a['Tema'].map(CORES_TEMAS)

                # 2. Gerar o gráfico SEM o parâmetro 'color=' para evitar barras fantasmas
                fig_p2a = px.bar(
                    df_plot_a, x='Proposições', y='Tema', orientation='h',
                    text_auto=True, template=template_grafico,
                    title="Total de proposições classificadas por eixo temático"
                )
                
                # 3. Pintar as barras diretamente e formatar os textos
                fig_p2a.update_traces(
                    marker_color=cores_a, 
                    textposition='outside', 
                    cliponaxis=False
                )
                
                # 4. Layout com altura fixa e barras grossas (bargap pequeno)
                fig_p2a.update_layout(
                    yaxis={'categoryorder': 'total ascending'}, 
                    height=500, 
                    bargap=0.15 
                )
                st.plotly_chart(fig_p2a, width='stretch')

            with tab_p2b:
                # Mesma lógica aplicada ao segundo gráfico
                df_plot_b = contagem_temas.sort_values('Deputados')
                cores_b = df_plot_b['Tema'].map(CORES_TEMAS)

                fig_p2b = px.bar(
                    df_plot_b, x='Deputados', y='Tema', orientation='h',
                    text_auto=True, template=template_grafico,
                    title="Número de deputados cujo tema principal de atuação é cada eixo"
                )
                
                fig_p2b.update_traces(
                    marker_color=cores_b, 
                    textposition='outside', 
                    cliponaxis=False
                )
                
                fig_p2b.update_layout(
                    yaxis={'categoryorder': 'total ascending'}, 
                    height=500, 
                    bargap=0.15
                )
                st.plotly_chart(fig_p2b, width='stretch')

            with tab_p2c:
                lista_deps_p2 = sorted(df_filtrado['txNomeParlamentar'].dropna().unique())
                dep_p2 = st.selectbox("Selecione o deputado:", lista_deps_p2, key='dep_p2')
                id_dep_p2 = None
                dep_p2_rows = df_filtrado[df_filtrado['txNomeParlamentar'] == dep_p2]
                if not dep_p2_rows.empty:
                    id_dep_p2 = dep_p2_rows['id_oficial'].iloc[0]
                if id_dep_p2:
                    perfil_ind = df_tc_filtrado[df_tc_filtrado['id_oficial'] == id_dep_p2]['tema'].value_counts().reset_index()
                    perfil_ind.columns = ['Tema', 'Proposições']
                    if perfil_ind.empty:
                        st.info(f"{dep_p2} não possui proposições classificadas nos eixos temáticos.")
                    else:
                        fig_ind_p2 = px.pie(
                            perfil_ind, values='Proposições', names='Tema',
                            color='Tema', color_discrete_map=CORES_TEMAS,
                            hole=0.4, template=template_grafico,
                            title=f"Distribuição temática — {dep_p2}"
                        )
                        st.plotly_chart(fig_ind_p2, width='stretch')

            with tab_p2d:
                tema_nuvem = st.selectbox("Selecione o eixo temático:", sorted(TEMAS_KEYWORDS.keys()), key='tema_nuvem_p2d')
                palavras_tema = df_tc_filtrado[df_tc_filtrado['tema'] == tema_nuvem]['palavras_chave'].dropna()
                
                if len(palavras_tema) == 0:
                    st.warning("Nenhuma palavra-chave encontrada para este tema com os filtros atuais.")
                else:
                    freq_palavras = obter_frequencia_palavras_chave(palavras_tema)
                    
                    if freq_palavras.empty:
                        st.info("Texto insuficiente para gerar a nuvem.")
                    else:
                        col_nuvem, col_tabela = st.columns([0.7, 0.3])
                        
                        with col_nuvem:
                            wc_tema = WordCloud(
                                width=900, height=500, background_color='white',
                                colormap='tab10', min_font_size=10, max_words=80
                            ).generate_from_frequencies(freq_palavras.to_dict())
                            
                            fig_nuvem_tema, ax_nt = plt.subplots(figsize=(12, 6))
                            ax_nt.imshow(wc_tema, interpolation='bilinear')
                            ax_nt.axis('off')
                            plt.tight_layout(pad=0)
                            st.pyplot(fig_nuvem_tema)
                            plt.close(fig_nuvem_tema)
                            st.caption(f"Nuvem gerada a partir de proposições classificadas como **{tema_nuvem}**")
                        
                        with col_tabela:
                            df_freq_tema = pd.DataFrame({'Palavra-chave': freq_palavras.index, 'Frequência': freq_palavras.values})
                            st.dataframe(df_freq_tema, hide_index=True, height=450, use_container_width=True)

    st.divider()


@st.fragment
def renderizar_p3():
    # --- 5c. P3 (🗳️ COMO O DEPUTADO VOTOU POR TEMA) ---
    st.header("🗳️ P3 — Como o Deputado Votou por Tema")

# 1. Cria o mapeamento de "Nome" para "Nome (Partido - Estado)" usando os dados já filtrados
    df_nomes_p3 = df_filtrado[['txNomeParlamentar', 'sgPartido', 'sgUF']].dropna().drop_duplicates(subset=['txNomeParlamentar'], keep='last')
    
    mapa_nomes_p3 = {}
    for _, row in df_nomes_p3.iterrows():
        nome = row['txNomeParlamentar']
        partido = row['sgPartido']
        uf = row['sgUF']
        mapa_nomes_p3[nome] = f"{nome} ({partido} - {uf})"

    # 2. Gera a lista com os nomes puros ordenados
    nomes_p3_disp = sorted(df_nomes_p3['txNomeParlamentar'].tolist())

    # 3. Monta as colunas e o selectbox com o format_func
    col_p3a, col_p3b = st.columns(2)
    
    with col_p3a:
        dep_p3 = st.selectbox(
            "Selecione o Deputado:", 
            options=nomes_p3_disp, 
            format_func=lambda x: mapa_nomes_p3.get(x, x), # <-- Muda apenas a exibição
            key='dep_p3'
        )
        
    with col_p3b:
        lista_temas = sorted(TEMAS_KEYWORDS.keys()) + ["Todos"]
        tema_p3 = st.selectbox("Selecione o Eixo Temático:", lista_temas, key='tema_p3')
        
    # --- FILTRO DE CATEGORIA ---
    categorias_disp = ["📋 Votação Principal", "📝 Alteração do Texto", "⚙️ Procedimento Legislativo", "🛑 Análise de Veto"]
    st.markdown("Filtrar por Categoria da Votação:")
    cols_cat = st.columns(len(categorias_disp))
    cat_selecionadas = []
    for i, cat in enumerate(categorias_disp):
        if cols_cat[i].checkbox(cat, value=True, key=f"chk_cat_{i}_{dep_p3}_{tema_p3}"):
            cat_selecionadas.append(cat)

    id_dep_p3 = None
    dep_rows = df_filtrado[df_filtrado['txNomeParlamentar'] == dep_p3]
    if not dep_rows.empty:
        id_dep_p3 = dep_rows['id_oficial'].iloc[0]

    if not id_dep_p3:
        return

    # Carrega votos só desse deputado (cacheado por deputado)
    df_raw = carregar_votos_deputado(str(id_dep_p3))
    df_votos_temas = pd.merge(df_raw, df_temas_classificado[['idProposicao_link', 'tema', 'ementa', 'nome_projeto']], on='idProposicao_link', how='inner')

    if df_votos_temas.empty:
        st.info(f"Sem registros de votos para **{dep_p3}** no período disponível.")
        return

    if tema_p3 == "Todos":
        votos_filtro = df_votos_temas.copy()
    else:
        votos_filtro = df_votos_temas[df_votos_temas['tema'] == tema_p3].copy()

    if votos_filtro.empty:
        st.info(f"Sem registros de votos de **{dep_p3}** em votações do eixo **{tema_p3}** no período disponível.")
        return

    _proc = 'requerimento|urgência|adiamento|retirada|inversão|dispensa|recurso|preferência|preferencia|ata'
    _alt  = 'destaque|emenda|substitutivo|redação final|mantido o texto|suprimido|votação em separado'
    _d = votos_filtro['descricao_votacao'].str.lower().fillna('')
    votos_filtro['categoria'] = "📋 Votação Principal"
    votos_filtro.loc[_d.str.contains(_alt),  'categoria'] = "📝 Alteração do Texto"
    votos_filtro.loc[_d.str.contains(_proc), 'categoria'] = "⚙️ Procedimento Legislativo"
    votos_filtro.loc[_d.str.contains('veto'), 'categoria'] = "🛑 Análise de Veto"

   
    votos_filtro = votos_filtro[votos_filtro['categoria'].isin(cat_selecionadas)]

    if votos_filtro.empty:
        st.warning("Nenhuma votação corresponde às categorias selecionadas.")
        return

    if 'hora_votacao' in votos_filtro.columns:
        votos_filtro['hora_votacao'] = pd.to_datetime(votos_filtro['hora_votacao'], errors='coerce')
        votos_filtro = votos_filtro.sort_values('hora_votacao', ascending=False)

    # --- FILTRO VISUAL DE SIM/NÃO ---
    st.markdown("Filtrar por Voto:")
    tipos_disponiveis = votos_filtro['voto'].dropna().unique().tolist()
    num_cols_voto = len(tipos_disponiveis) if tipos_disponiveis else 1
    cols_voto = st.columns(num_cols_voto)
    tipos_selecionados = []
    for i, tipo in enumerate(tipos_disponiveis):
        tipo_str = str(tipo)
        if tipo_str == 'Sim': label = "🟢 Sim"
        elif tipo_str == 'Não': label = "🔴 Não"
        else: label = f"⚪ {tipo_str}"
        if cols_voto[i].checkbox(label, value=True, key=f"chk_voto_{tipo_str}_{dep_p3}_{tema_p3}"):
            tipos_selecionados.append(tipo_str)

    votos_exibicao = votos_filtro[votos_filtro['voto'].isin(tipos_selecionados)]

    def formatar_data_hora(valor_dt):
        if pd.notnull(valor_dt):
            return f"🕒 {valor_dt.strftime('%d/%m/%Y às %H:%M')} — "
        return ""

    if votos_exibicao.empty:
        st.warning("Nenhum voto corresponde aos tipos selecionados no filtro.")
        return

    st.caption(f"Exibindo {len(votos_exibicao)} votação(ões) correspondente(s) aos filtros.")

    # --- PREPARAÇÃO PARA ANÁLISE DE COMPORTAMENTO ---
    partido_dep = dep_rows['sgPartido'].iloc[0]
    bancada_maioria = {}
    if 'sgPartido' in df_votos_contagem.columns:
        votos_bancada = df_votos_contagem[df_votos_contagem['sgPartido'] == partido_dep]
        if not votos_bancada.empty:
            idx_max = votos_bancada.groupby('id_votacao_str')['perc'].idxmax()
            bancada_maioria = votos_bancada.loc[idx_max].set_index('id_votacao_str')['voto'].to_dict()

    # --- PRÉ-PROCESSAMENTO DAS MÉTRICAS ---
    total_votos = len(votos_exibicao)
    vitorias = manobras_total = fuga_total = obstrucao_total = rebeldia_total = votos_comparaveis_partido = 0
    cards_processados = []

    for _, row in votos_exibicao.iterrows():
        voto_atual = str(row['voto'])
        voto_lower = voto_atual.lower()
        id_vot = str(row.get('id_votacao_str', ''))
        data_formatada = formatar_data_hora(row.get('hora_votacao'))
        descricao = str(row.get('descricao_votacao', ''))
        ementa = str(row.get('ementa', ''))
        tipo_badge = str(row['categoria'])
        nome_proj = row.get('nome_projeto')
        if pd.isna(nome_proj) or str(nome_proj).strip() in ['', 'nan', 'None']:
            nome_proj = f"ID: {row.get('idProposicao_link', 'Desconhecido')}"
        else:
            nome_proj = str(nome_proj)

        aprovado_val = row.get('resultado_aprovado')
        desc_texto_lower = str(descricao).lower().strip()
        if pd.isna(aprovado_val) or str(aprovado_val).strip() in ['', 'nan', 'None']:
            if 'mantido o texto' in desc_texto_lower: desfecho = "🛡️ **Texto Base Mantido**"
            elif desc_texto_lower.startswith('aprovad') or 'aprovado ' in desc_texto_lower: desfecho = "✅ **Aprovado**"
            elif desc_texto_lower.startswith('rejeitad') or 'rejeitado ' in desc_texto_lower: desfecho = "❌ **Rejeitado**"
            else: desfecho = "⚠️ N/D"
        elif str(aprovado_val).strip() in ['1', '1.0', 'True', 'true']: desfecho = "✅ **Aprovado**"
        elif str(aprovado_val).strip() in ['0', '0.0', 'False', 'false']: desfecho = "❌ **Rejeitado**"
        else: desfecho = "⚠️ N/D"

        if voto_atual == 'Sim' and ('Aprovado' in desfecho or 'Mantido' in desfecho): vitorias += 1
        elif voto_atual == 'Não' and 'Rejeitado' in desfecho: vitorias += 1
        if "Votação Principal" not in tipo_badge: manobras_total += 1
        if voto_lower in ['abstenção', 'abstencao', 'art. 17', 'ausente', 'branco']: fuga_total += 1
        elif voto_lower in ['obstrução', 'obstrucao']: obstrucao_total += 1

        is_rebelde = False
        voto_partido = None
        if voto_atual in ['Sim', 'Não'] and id_vot in bancada_maioria:
            voto_partido = bancada_maioria[id_vot]
            if voto_partido in ['Sim', 'Não']:
                votos_comparaveis_partido += 1
                if voto_atual != voto_partido:
                    rebeldia_total += 1
                    is_rebelde = True

        cards_processados.append({
            'nome_proj': nome_proj, 'tipo_badge': tipo_badge, 'data_formatada': data_formatada,
            'descricao': descricao, 'voto_atual': voto_atual, 'desfecho': desfecho, 'ementa': ementa,
            'is_rebelde': is_rebelde, 'voto_partido': voto_partido
        })

    # --- DESENHO DA TIMELINE ---
    # --- DESENHO DA TIMELINE COM PAGINAÇÃO ---
    CARDS_POR_PAGINA = 20
    total_cards = len(cards_processados)
    total_paginas = max(1, -(-total_cards // CARDS_POR_PAGINA))  # divisão com teto

    chave_pag = f"pag_p3_{dep_p3}_{tema_p3}"
    if chave_pag not in st.session_state:
        st.session_state[chave_pag] = 0

    pagina_atual = st.session_state[chave_pag]

    # Garante que a página não fique fora do range quando o filtro muda
    if pagina_atual >= total_paginas:
        st.session_state[chave_pag] = 0
        pagina_atual = 0

    inicio = pagina_atual * CARDS_POR_PAGINA
    fim = min(inicio + CARDS_POR_PAGINA, total_cards)
    cards_pagina = cards_processados[inicio:fim]

    st.caption(f"Exibindo {inicio+1}–{fim} de {total_cards} votações | Página {pagina_atual+1} de {total_paginas}")

    with st.container(height=650):
        for card in cards_pagina:
            with st.container(border=True):
                col_texto, col_votos = st.columns([0.75, 0.25])
                with col_texto:
                    st.markdown(f"**🏛️ {card['nome_proj']}** • {card['tipo_badge']}")
                    st.caption(f"{card['data_formatada']} — {card['descricao']}")
                with col_votos:
                    cor_voto = "🟢" if card['voto_atual'] == 'Sim' else ("🔴" if card['voto_atual'] == 'Não' else "⚪")
                    texto_voto_seguro = str(card['voto_atual']).upper()
                    if card.get('is_rebelde'):
                        st.markdown(f"Deputado: {cor_voto} **{texto_voto_seguro}** 🏴‍☠️")
                        cor_partido = "🟢" if card['voto_partido'] == 'Sim' else "🔴"
                        st.caption(f"*(Partido: {cor_partido} {card['voto_partido'].upper()})*")
                    else:
                        st.markdown(f"Deputado: {cor_voto} **{texto_voto_seguro}**")
                    st.markdown(f"Plenário: {card['desfecho']}")
                with st.expander("Ver Ementa"):
                    st.caption(card['ementa'])

    # Botões de navegação
    col_ant, col_info, col_prox = st.columns([0.2, 0.6, 0.2])
    with col_ant:
        if st.button("← Anterior", disabled=(pagina_atual == 0), key=f"btn_ant_{chave_pag}"):
            st.session_state[chave_pag] -= 1
            st.rerun()
    with col_info:
        st.markdown(f"<div style='text-align:center; padding-top:6px'>Página {pagina_atual+1} de {total_paginas}</div>", unsafe_allow_html=True)
    with col_prox:
        if st.button("Próxima →", disabled=(pagina_atual >= total_paginas - 1), key=f"btn_prox_{chave_pag}"):
            st.session_state[chave_pag] += 1
            st.rerun()

    # --- ANÁLISE DE COMPORTAMENTO ---
    st.markdown("#### 🔍 Análise de Comportamento")
    st.caption("Resumo analítico baseado nas votações listadas")
    perc_vitorias = (vitorias / total_votos) * 100 if total_votos > 0 else 0
    perc_manobra = (manobras_total / total_votos) * 100 if total_votos > 0 else 0
    perc_fuga = (fuga_total / total_votos) * 100 if total_votos > 0 else 0
    perc_obstrucao = (obstrucao_total / total_votos) * 100 if total_votos > 0 else 0
    perc_rebeldia = (rebeldia_total / votos_comparaveis_partido) * 100 if votos_comparaveis_partido > 0 else 0

    c1, c2 = st.columns(2)
    c1.metric("🎯 Alinhamento c/ Plenário", f"{perc_vitorias:.0f}%", f"{vitorias} vitórias em {total_votos} votos", delta_color="off")
    c2.metric("⚙️ Esforço em Bastidores", f"{perc_manobra:.0f}%", "Votos em manobras/alterações", delta_color="off")
    st.write("")
    c4, c5, c6 = st.columns(3)
    c4.metric("🫣 Taxa de Fuga (Omissão)", f"{perc_fuga:.0f}%", f"{fuga_total} abstenções ou ausências", delta_color="off")
    c5.metric("🛑 Obstrução Direta", f"{perc_obstrucao:.0f}%", f"{obstrucao_total} votos de obstrução oficial", delta_color="off")
    subtitulo_c6 = f"{rebeldia_total} voto(s) em {votos_comparaveis_partido} votações c/ o partido" if votos_comparaveis_partido > 0 else "Sem votos nominais comparáveis"
    c6.metric("🏴‍☠️ Rebeldia Temática", f"{perc_rebeldia:.0f}% divergente" if votos_comparaveis_partido > 0 else "N/A", subtitulo_c6, delta_color="off")

    st.divider()


def renderizar_p4():
    # --- 6. P4 (PERFIL POR ESCOLARIDADE + DETALHAMENTO) ---
    st.header("🎓 P4 — Perfil por Escolaridade")

    df_u = df_filtrado.drop_duplicates(subset=['id_oficial'])
    esc_data = df_u['escolaridade'].value_counts().reset_index()
    esc_data.columns = ['Escolaridade', 'Qtd']

    fig_p4 = px.bar(
        esc_data, x='Qtd', y='Escolaridade', 
        orientation='h', color='Qtd', color_continuous_scale='Blues', 
        text_auto=True, template=template_grafico,
        labels={'Qtd': 'Número de Deputados'}
    )
    fig_p4.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_p4, width='stretch')

    st.subheader("🔍 Ver Deputados por Nível de Escolaridade")
    escolaridades_disponiveis = sorted(df_u['escolaridade'].dropna().unique())

    if escolaridades_disponiveis:
        nivel_sel = st.selectbox("Selecione um nível de instrução para listar os parlamentares:", escolaridades_disponiveis)
        
        # 1. Filtra, seleciona e ordena
        df_nivel_sel = df_u[df_u['escolaridade'] == nivel_sel][['txNomeParlamentar', 'sgPartido', 'sgUF']].sort_values(by='txNomeParlamentar')
        
        # 2. Renomeia as colunas apenas para a exibição
        df_nivel_sel = df_nivel_sel.rename(columns={
            'txNomeParlamentar': 'Nome', 
            'sgPartido': 'Partido', 
            'sgUF': 'Estado'
        })
        
        st.markdown(f"**Parlamentares com '{nivel_sel}' ({len(df_nivel_sel)} encontrados):**")
        st.dataframe(df_nivel_sel, width='stretch', hide_index=True)
    else:
        st.info("Nenhum dado de escolaridade disponível para os filtros atuais.")

    st.divider()


def renderizar_p6():
    # =============================================================================
    # --- 10. P6 — ESCOLARIDADE × (GASTOS / FIDELIDADE / PROPOSIÇÕES / PRESENÇA) ---
    # =============================================================================
    st.header("🎓 P6 — Escolaridade em Perspectiva")
    st.markdown(
        "Correlação entre o nível de instrução do deputado e quatro dimensões da sua atuação: "
        "gastos, fidelidade partidária, produção legislativa e presença."
    )

    # ------------------------------------------------------------------
    # BASE COMPARTILHADA: perfil por deputado (1 linha por deputado)
    # ------------------------------------------------------------------
    perfil_dep_esc = (
        df_filtrado[['id_oficial', 'txNomeParlamentar', 'escolaridade', 'sgPartido', 'sgUF']]
        .drop_duplicates(subset=['id_oficial'])
    )

    # Ordem pedagógica dos níveis de escolaridade — usada em todos os sub-gráficos
# Ordem pedagógica dos níveis de escolaridade (baseado no Observatório Político)
    ORDEM_ESC = [
        'Não Informado',
        'Primário Incompleto',
        'Ensino Fundamental Incompleto',
        'Ensino Fundamental',
        'Secundário Incompleto',
        'Ensino Médio Incompleto',
        'Secundário',
        'Ensino Médio',
        'Ensino Técnico',
        'Superior Incompleto',
        'Superior',
        'Especialização',
        'Pós-Graduação',
        'Mestrado Incompleto',
        'Mestrado',
        'Doutorado Incompleto',
        'Doutorado'
    ]
    
    niveis_presentes = [n for n in ORDEM_ESC if n in perfil_dep_esc['escolaridade'].unique()]
    niveis_extras = [
        n for n in perfil_dep_esc['escolaridade'].dropna().unique()
        if n not in ORDEM_ESC
    ]
    ordem_final = niveis_presentes + niveis_extras
    

    def aplicar_ordem(df, col='escolaridade'):
        df = df.copy()
        df = df[df[col].notna()]   # <-- remove linhas com escolaridade nula
        df[col] = pd.Categorical(df[col], categories=ordem_final, ordered=True)
        return df.sort_values(col)

    def grafico_media_boxplot(df_ind, col_y, label_y, key_prefix, cor_escala='Blues'):
        """Reutilizável: gera aba de média + aba de boxplot para qualquer métrica × escolaridade."""
        tab_med, tab_box = st.tabs(["📊 Média por Nível", "📦 Distribuição (Boxplot)"])

        media = (
            df_ind.groupby('escolaridade')[col_y]
            .agg(['mean', 'count'])
            .reset_index()
            .rename(columns={'mean': 'media', 'count': 'n'})
        )
        media = aplicar_ordem(media)

        with tab_med:
            fig_m = px.bar(
                media, x='escolaridade', y='media',
                color='media', color_continuous_scale=cor_escala,
                text_auto=',.1f',
                hover_data=['n'],
                labels={'escolaridade': 'Escolaridade', 'media': label_y, 'n': 'Nº Deputados'},
                template=template_grafico,
            )
            # <-- AQUI: Adicionado categoryorder e categoryarray no xaxis
            fig_m.update_layout(
                xaxis_tickangle=-30, 
                coloraxis_showscale=False,
                xaxis={'categoryorder': 'array', 'categoryarray': ordem_final}
            )
            fig_m.update_traces(textposition='outside')
            st.plotly_chart(fig_m, width='stretch')
            st.caption("Passe o mouse sobre as barras para ver o número de deputados em cada nível.")

        with tab_box:
            df_box = aplicar_ordem(df_ind)
            fig_b = px.box(
                df_box, x='escolaridade', y=col_y,
                color='escolaridade', points='outliers',
                labels={'escolaridade': 'Escolaridade', col_y: label_y},
                template=template_grafico,
            )
            # <-- AQUI: Adicionado categoryorder e categoryarray no xaxis
            fig_b.update_layout(
                xaxis_tickangle=-30, 
                showlegend=False,
                xaxis={'categoryorder': 'array', 'categoryarray': ordem_final}
            )
            st.plotly_chart(fig_b, width='stretch')

    # ------------------------------------------------------------------
    # FUNÇÃO NOVA: presença individual por deputado (P6d)
    # ------------------------------------------------------------------
    @st.cache_data
    def carregar_presenca_individual():
        conn = sqlite3.connect('banco.db')

        # Sessões Deliberativas (plenário)
        q_plenario = """
        SELECT p.dep_id AS id_oficial, COUNT(p.evt_id) AS presencas_plenario
        FROM PresencaDeputado p
        JOIN Evento e ON p.evt_id = e.evt_id
        WHERE e.evt_tipo = 'Sessão Deliberativa'
        GROUP BY p.dep_id
        """
        df_pl = pd.read_sql_query(q_plenario, conn)
        df_pl['id_oficial'] = df_pl['id_oficial'].astype(str).str.split('.').str[0]

        # Todos os eventos (qualquer tipo)
        q_todos = """
        SELECT dep_id AS id_oficial, COUNT(evt_id) AS presencas_total
        FROM PresencaDeputado
        GROUP BY dep_id
        """
        df_todos = pd.read_sql_query(q_todos, conn)
        df_todos['id_oficial'] = df_todos['id_oficial'].astype(str).str.split('.').str[0]

        conn.close()

        df_pres_ind = pd.merge(df_todos, df_pl, on='id_oficial', how='left')
        df_pres_ind['presencas_plenario'] = df_pres_ind['presencas_plenario'].fillna(0).astype(int)
        # Taxa: % das presenças totais que foram em plenário
        df_pres_ind['taxa_plenario'] = (
            df_pres_ind['presencas_plenario'] / df_pres_ind['presencas_total'] * 100
        ).round(1)
        return df_pres_ind

    df_pres_individual = carregar_presenca_individual()

    # ------------------------------------------------------------------
    # ABAS PRINCIPAIS DE P6
    # ------------------------------------------------------------------
    tab_p6a, tab_p6b, tab_p6c, tab_p6d = st.tabs([
        "💰 a) Gastos",
        "🤝 b) Fidelidade Partidária",
        "📜 c) Proposições",
        "📅 d) Presença",
    ])

    # ── P6a: Escolaridade × Gastos ─────────────────────────────────────
    with tab_p6a:
        st.subheader("Escolaridade × Gasto médio da cota parlamentar")

        gastos_por_dep = (
            df_filtrado.groupby('id_oficial')['vlrLiquido']
            .sum().reset_index()
            .rename(columns={'vlrLiquido': 'total_gasto'})
        )
        df_esc_gastos = pd.merge(perfil_dep_esc, gastos_por_dep, on='id_oficial', how='inner')

        if df_esc_gastos.empty:
            st.info("Sem dados suficientes.")
        else:
            grafico_media_boxplot(df_esc_gastos, 'total_gasto', 'Gasto Médio (R$)', 'p6a', 'Blues')

            with st.expander("📋 Ver tabela detalhada"):
                df_tab = df_esc_gastos[['txNomeParlamentar','sgPartido','sgUF','escolaridade','total_gasto']].copy()
                df_tab = df_tab.sort_values('total_gasto', ascending=False).reset_index(drop=True)
                df_tab.columns = ['Deputado','Partido','UF','Escolaridade','Gasto Total (R$)']
                df_tab['Gasto Total (R$)'] = df_tab['Gasto Total (R$)'].map('R$ {:,.2f}'.format)
                esc_sel_a = st.multiselect("Filtrar escolaridade:", ordem_final, default=[], key='esc_tab_a')
                df_exib_a = df_tab if not esc_sel_a else df_tab[df_tab['Escolaridade'].isin(esc_sel_a)]
                st.dataframe(df_exib_a, hide_index=True, use_container_width=True, height=380)

    # ── P6b: Escolaridade × Fidelidade Partidária ──────────────────────
    with tab_p6b:
        st.subheader("Escolaridade × Fidelidade à bancada")
        st.caption(
            "Fidelidade = % de votações em que o deputado votou com a maioria do seu partido. "
            "Calculada individualmente a partir dos dados de alinhamento por partido (P10)."
        )

        # df_alinhamento está por partido; precisamos descer para o nível do deputado.
        # Usamos df_votos_contagem que já tem sgPartido + id_votacao_str.
        # A fidelidade individual exige recalcular — fazemos isso aqui com cache.

        fidelidade_dep = calcular_fidelidade_individual()

        ids_no_filtro_p6 = df_filtrado['id_oficial'].unique()
        df_esc_fid = pd.merge(perfil_dep_esc, fidelidade_dep, on='id_oficial', how='inner')
        df_esc_fid = df_esc_fid[df_esc_fid['id_oficial'].isin(ids_no_filtro_p6)]
        

        # Filtra apenas deputados no filtro global
        ids_no_filtro_p6 = df_filtrado['id_oficial'].unique()
        df_esc_fid = pd.merge(perfil_dep_esc, fidelidade_dep, on='id_oficial', how='inner')
        df_esc_fid = df_esc_fid[df_esc_fid['id_oficial'].isin(ids_no_filtro_p6)]

        if df_esc_fid.empty:
            st.info("Sem dados de votações suficientes para calcular fidelidade individual.")
        else:
            grafico_media_boxplot(df_esc_fid, 'fidelidade', 'Fidelidade Média (%)', 'p6b', 'Purples')

            with st.expander("📋 Ver tabela detalhada"):
                df_tab_b = df_esc_fid[['txNomeParlamentar','sgPartido','sgUF','escolaridade','fidelidade']].copy()
                df_tab_b = df_tab_b.sort_values('fidelidade', ascending=False).reset_index(drop=True)
                df_tab_b.columns = ['Deputado','Partido','UF','Escolaridade','Fidelidade (%)']
                esc_sel_b = st.multiselect("Filtrar escolaridade:", ordem_final, default=[], key='esc_tab_b')
                df_exib_b = df_tab_b if not esc_sel_b else df_tab_b[df_tab_b['Escolaridade'].isin(esc_sel_b)]
                st.dataframe(df_exib_b, hide_index=True, use_container_width=True, height=380)

    # ── P6c: Escolaridade × Nº de Proposições ─────────────────────────
    with tab_p6c:
        st.subheader("Escolaridade × Número de proposições assinadas")

        # df_autores: id_oficial + idProposicao_link (já carregado globalmente)
        props_por_dep = (
            df_autores.groupby('id_oficial')['idProposicao_link']
            .count().reset_index(name='n_proposicoes')
        )
        df_esc_prop = pd.merge(perfil_dep_esc, props_por_dep, on='id_oficial', how='left')
        df_esc_prop['n_proposicoes'] = df_esc_prop['n_proposicoes'].fillna(0).astype(int)
        df_esc_prop = df_esc_prop[df_esc_prop['id_oficial'].isin(ids_no_filtro_p6)]

        if df_esc_prop.empty:
            st.info("Sem dados de proposições para o filtro atual.")
        else:
            grafico_media_boxplot(df_esc_prop, 'n_proposicoes', 'Média de Proposições Assinadas', 'p6c', 'Greens')

            with st.expander("📋 Ver tabela detalhada"):
                df_tab_c = df_esc_prop[['txNomeParlamentar','sgPartido','sgUF','escolaridade','n_proposicoes']].copy()
                df_tab_c = df_tab_c.sort_values('n_proposicoes', ascending=False).reset_index(drop=True)
                df_tab_c.columns = ['Deputado','Partido','UF','Escolaridade','Nº Proposições']
                esc_sel_c = st.multiselect("Filtrar escolaridade:", ordem_final, default=[], key='esc_tab_c')
                df_exib_c = df_tab_c if not esc_sel_c else df_tab_c[df_tab_c['Escolaridade'].isin(esc_sel_c)]
                st.dataframe(df_exib_c, hide_index=True, use_container_width=True, height=380)

    # ── P6d: Escolaridade × Presença ───────────────────────────────────
    with tab_p6d:
        st.subheader("Escolaridade × Presença em eventos e no plenário")
        st.caption(
            "**Presença total** = todos os eventos registrados no sistema (comissões, audiências, etc.). "
            "**Plenário** = apenas Sessões Deliberativas. "
            "**Taxa plenário** = % das presenças totais que foram em sessões deliberativas."
        )

        df_esc_pres = pd.merge(perfil_dep_esc, df_pres_individual, on='id_oficial', how='left')
        df_esc_pres['presencas_total']   = df_esc_pres['presencas_total'].fillna(0).astype(int)
        df_esc_pres['presencas_plenario'] = df_esc_pres['presencas_plenario'].fillna(0).astype(int)
        df_esc_pres['taxa_plenario']     = df_esc_pres['taxa_plenario'].fillna(0)
        df_esc_pres = df_esc_pres[df_esc_pres['id_oficial'].isin(ids_no_filtro_p6)]

        if df_esc_pres.empty:
            st.info("Sem dados de presença para o filtro atual.")
        else:
            metrica_p6d = st.radio(
                "Qual métrica analisar?",
                ["Presença total (todos eventos)", "Presença no plenário (deliberativas)", "Taxa plenário (%)"],
                horizontal=True,
                key='radio_p6d'
            )

            col_map = {
                "Presença total (todos eventos)":          ('presencas_total',    'Média de Presenças Totais',        'Oranges'),
                "Presença no plenário (deliberativas)":    ('presencas_plenario', 'Média de Presenças no Plenário',   'Reds'),
                "Taxa plenário (%)":                       ('taxa_plenario',      'Taxa Média Plenário (%)',           'YlOrRd'),
            }
            col_y_d, label_y_d, cor_d = col_map[metrica_p6d]

            grafico_media_boxplot(df_esc_pres, col_y_d, label_y_d, 'p6d', cor_d)

            with st.expander("📋 Ver tabela detalhada"):
                df_tab_d = df_esc_pres[[
                    'txNomeParlamentar','sgPartido','sgUF','escolaridade',
                    'presencas_total','presencas_plenario','taxa_plenario'
                ]].copy().sort_values('presencas_total', ascending=False).reset_index(drop=True)
                df_tab_d.columns = ['Deputado','Partido','UF','Escolaridade',
                                    'Presenças Totais','Presenças Plenário','Taxa Plenário (%)']
                esc_sel_d = st.multiselect("Filtrar escolaridade:", ordem_final, default=[], key='esc_tab_d')
                df_exib_d = df_tab_d if not esc_sel_d else df_tab_d[df_tab_d['Escolaridade'].isin(esc_sel_d)]
                st.dataframe(df_exib_d, hide_index=True, use_container_width=True, height=380)

    st.divider()

def renderizar_p8():
    # =============================================================================
    # --- 11. P8 — INFLUÊNCIA: % DE PROPOSTAS APROVADAS POR DEPUTADO ---
    # =============================================================================
    st.header("🏆 P8 — Influência Legislativa: Propostas Aprovadas")
    st.markdown(
        "Mede a **efetividade legislativa** de cada deputado: qual percentual das proposições "
        "que ele assinou foi aprovada, e qual sua participação no total de aprovações do período."
    )


    df_influencia = calcular_influencia_p8()

    # Junta com nome/partido/UF/escolaridade do df_principal
    perfil_deps = df_principal[['id_oficial', 'txNomeParlamentar', 'sgPartido', 'sgUF']].drop_duplicates(subset=['id_oficial'])
    df_inf_completo = pd.merge(df_influencia, perfil_deps, on='id_oficial', how='inner')

    # Aplica filtro global (apenas deputados que estão no df_filtrado)
    ids_no_filtro = df_filtrado['id_oficial'].unique()
    df_inf_filtrado = df_inf_completo[df_inf_completo['id_oficial'].isin(ids_no_filtro)].copy()

    if df_inf_filtrado.empty:
        st.info("Sem dados de proposições/votações para calcular influência com os filtros atuais.")
    else:
        tab_p8_taxa, tab_p8_abs, tab_p8_scatter = st.tabs([
            "🎯 Taxa de Aprovação (%)", "📌 Nº Absoluto de Aprovações", "🔵 Visão Geral (Scatter)"
        ])

        col_n_p8, _ = st.columns([0.3, 0.7])
        with col_n_p8:
            n_p8 = st.slider("Quantos deputados exibir no ranking?", 5, 50, 20, key='sl_p8')

        with tab_p8_taxa:
            st.subheader("Deputados com maior taxa de aprovação das proposições assinadas")
            st.caption("Mínimo de 3 proposições assinadas para entrar no ranking (evita distorção com quem só assinou 1).")

            # Adicionado .copy() para poder criar a nova coluna com segurança
            df_p8_taxa = df_inf_filtrado[df_inf_filtrado['total_proposicoes'] >= 3].sort_values('taxa_aprovacao', ascending=False).head(n_p8).copy()

            if df_p8_taxa.empty:
                st.info("Nenhum deputado com 3+ proposições no filtro atual.")
            else:
                # Criando a coluna formatada
                df_p8_taxa['NomeExibicao'] = df_p8_taxa['txNomeParlamentar'] + " (" + df_p8_taxa['sgPartido'] + " - " + df_p8_taxa['sgUF'] + ")"

                fig_p8_t = px.bar(
                    df_p8_taxa,
                    x='taxa_aprovacao', 
                    y='NomeExibicao', # <-- Usando a nova coluna
                    orientation='h',
                    color='taxa_aprovacao', color_continuous_scale='Greens',
                    labels={
                        'taxa_aprovacao': 'Taxa de Aprovação (%)',
                        'NomeExibicao': 'Deputado' # <-- Label atualizado
                    },
                    # Ajustando o hover para esconder o NomeExibicao e mostrar os dados separados
                    hover_data={'NomeExibicao': False, 'txNomeParlamentar': True, 'sgPartido': True, 'sgUF': True, 'total_proposicoes': True, 'proposicoes_aprovadas': True},
                    template=template_grafico,
                    text_auto='.1f'
                )
                fig_p8_t.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_p8_t, width='stretch')

        with tab_p8_abs:
            st.subheader("Deputados com maior número absoluto de proposições aprovadas")
            st.caption("Deputados que mais contribuíram com o volume total de aprovações no período.")

            # Adicionado .copy()
            df_p8_abs = df_inf_filtrado.sort_values('proposicoes_aprovadas', ascending=False).head(n_p8).copy()
            
            # Criando a coluna formatada
            df_p8_abs['NomeExibicao'] = df_p8_abs['txNomeParlamentar'] + " (" + df_p8_abs['sgPartido'] + " - " + df_p8_abs['sgUF'] + ")"

            fig_p8_a = px.bar(
                df_p8_abs,
                x='proposicoes_aprovadas', 
                y='NomeExibicao', # <-- Usando a nova coluna
                orientation='h',
                color='perc_do_total_aprov', color_continuous_scale='Teal',
                labels={
                    'proposicoes_aprovadas': 'Proposições Aprovadas',
                    'NomeExibicao': 'Deputado', # <-- Label atualizado
                    'perc_do_total_aprov': '% do Total de Aprovações'
                },
                # Ajustando o hover
                hover_data={'NomeExibicao': False, 'txNomeParlamentar': True, 'sgPartido': True, 'sgUF': True, 'taxa_aprovacao': True, 'perc_do_total_aprov': True},
                template=template_grafico,
                text_auto=True
            )
            fig_p8_a.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                coloraxis_colorbar_title='% do Total'
            )
            st.plotly_chart(fig_p8_a, width='stretch')

        with tab_p8_scatter:
            st.subheader("Volume de proposições assinadas × taxa de aprovação")
            st.caption("Cada ponto é um deputado. Tamanho do ponto = número de aprovações absolutas.")

            fig_p8_s = px.scatter(
                df_inf_filtrado[df_inf_filtrado['total_proposicoes'] >= 3],
                x='total_proposicoes',
                y='taxa_aprovacao',
                size='proposicoes_aprovadas',
                color='sgPartido',
                hover_name='txNomeParlamentar',
                hover_data={'sgUF': True, 'proposicoes_aprovadas': True, 'perc_do_total_aprov': True},
                labels={
                    'total_proposicoes': 'Total de Proposições Assinadas',
                    'taxa_aprovacao': 'Taxa de Aprovação (%)',
                    'sgPartido': 'Partido'
                },
                template=template_grafico,
                size_max=30,
            )
            fig_p8_s.update_layout(legend_title='Partido')
            st.plotly_chart(fig_p8_s, width='stretch')
            st.caption(
                "Deputados no canto superior direito assinam muitas proposições E têm alta taxa de aprovação "
                "— são os mais influentes legislativamente."
            )

    st.divider()

def renderizar_p10():
    # --- 7. P10 (🤝 ALINHAMENTO INTERNO DOS PARTIDOS) ---
    st.header("🤝 P10 — Alinhamento Interno dos Partidos (Fidelidade à Bancada)")

    partidos_no_filtro = df_filtrado['sgPartido'].unique()
    df_alinhamento_filtrado = df_alinhamento[df_alinhamento['sgPartido'].isin(partidos_no_filtro)].copy()

    if df_alinhamento_filtrado.empty:
        st.warning("⚠️ Dados de votações insuficientes para os partidos selecionados ou não contêm votos 'Sim'/'Não'. Impossível calcular o alinhamento interno (P10) no momento.")
    else:
        tab_p10_geral, tab_p10_busca, tab_p10_divididas = st.tabs([
            "📊 Alinhamento Geral",
            "🔍 Buscar Votação",
            "⚡ Mais Divididas"
        ])

        with tab_p10_geral:
            st.markdown("Porcentagem média de vezes que os deputados de cada partido votaram junto com a maioria da sua própria bancada (em votações Sim/Não).")
            df_alinhamento_filtrado.sort_values(by='perc_alinhamento', ascending=False, inplace=True)
            fig_p10 = px.bar(
                df_alinhamento_filtrado,
                x='sgPartido', y='perc_alinhamento',
                color='perc_alinhamento', color_continuous_scale='Cividis',
                labels={'perc_alinhamento': 'Alinhamento Médio (%)', 'sgPartido': 'Partido'},
                template=template_grafico,
                text_auto='.1f'
            )
            fig_p10.update_layout(xaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig_p10, width='stretch')

        with tab_p10_busca:
            st.markdown("Busque pelo tema ou descrição de uma votação e veja como cada partido votou.")

            if df_votos_contagem.empty:
                st.info("Dados de votações não disponíveis para a busca.")
            else:
                busca_tema = st.text_input("🔎 Buscar votação por tema/descrição:", placeholder="Ex: reforma, imposto, educação...")
                df_contagem_filtrada = df_votos_contagem[df_votos_contagem['sgPartido'].isin(partidos_no_filtro)].copy()
                opcoes_votacao = df_contagem_filtrada[['id_votacao_str', 'tema_label']].drop_duplicates(subset='id_votacao_str')

                if busca_tema:
                    mask_busca = opcoes_votacao['tema_label'].str.contains(busca_tema, case=False, na=False)
                    opcoes_votacao = opcoes_votacao[mask_busca]

                if opcoes_votacao.empty:
                    st.warning("Nenhuma votação encontrada para esse termo.")
                else:
                    opcoes_votacao = opcoes_votacao.copy()
                    opcoes_votacao['label_curto'] = opcoes_votacao['tema_label'].str[:120]
                    mapa_label_id = dict(zip(opcoes_votacao['label_curto'], opcoes_votacao['id_votacao_str']))

                    votacao_escolhida_label = st.selectbox(
                        f"{len(opcoes_votacao)} votação(ões) encontrada(s) — selecione uma:",
                        options=list(mapa_label_id.keys())
                    )
                    votacao_escolhida_id = mapa_label_id[votacao_escolhida_label]
                    df_vot_sel = df_contagem_filtrada[df_contagem_filtrada['id_votacao_str'] == votacao_escolhida_id].copy()

                    if not df_vot_sel.empty:
                        fig_busca = px.bar(
                            df_vot_sel,
                            x='sgPartido', y='perc', color='voto',
                            color_discrete_map={'Sim': '#2ecc71', 'Não': '#e74c3c'},
                            barmode='stack',
                            labels={'perc': '% de Votos', 'sgPartido': 'Partido', 'voto': 'Voto'},
                            template=template_grafico,
                            text_auto='.0f'
                        )
                        fig_busca.update_layout(yaxis_title='% de Votos', legend_title='Voto', xaxis={'categoryorder': 'total descending'})
                        st.plotly_chart(fig_busca, width='stretch')

                        resumo = df_vot_sel.loc[df_vot_sel.groupby('sgPartido')['perc'].idxmax()][['sgPartido', 'voto', 'perc']].copy()
                        resumo.columns = ['Partido', 'Voto Majoritário', '% Maioria']
                        resumo['% Maioria'] = resumo['% Maioria'].round(1)
                        resumo.sort_values('Partido', inplace=True)
                        st.markdown("**Voto majoritário por partido nessa votação:**")
                        st.dataframe(resumo, width='stretch', hide_index=True)

        with tab_p10_divididas:
            st.markdown("Votações onde os partidos tiveram **menor coesão interna** — mais deputados do mesmo partido votando de formas diferentes.")

            if df_divisao.empty:
                st.info("Dados de divisão não disponíveis.")
            else:
                col_n, _ = st.columns([0.3, 0.7])
                with col_n:
                    n_divididas = st.slider("Quantas votações exibir?", 5, 30, 10)

                ids_com_dados = df_votos_contagem[df_votos_contagem['sgPartido'].isin(partidos_no_filtro)]['id_votacao_str'].unique()
                df_div_filtrada = df_divisao[df_divisao['id_votacao_str'].isin(ids_com_dados)].head(n_divididas).copy()

                if df_div_filtrada.empty:
                    st.warning("Nenhuma votação encontrada para os partidos selecionados.")
                else:
                    df_div_filtrada['label_curto'] = df_div_filtrada['tema_label'].str[:60] + '...'

                    fig_div = px.bar(
                        df_div_filtrada,
                        x='coesao_media', y='label_curto',
                        orientation='h',
                        color='coesao_media', color_continuous_scale='RdYlGn',
                        labels={'coesao_media': 'Coesão Média dos Partidos (%)', 'label_curto': 'Votação'},
                        template=template_grafico,
                        text_auto='.1f'
                    )
                    fig_div.update_layout(yaxis={'categoryorder': 'total ascending'}, height=150 + (n_divididas * 35), coloraxis_colorbar_title='Coesão (%)')
                    st.plotly_chart(fig_div, width='stretch')

                    st.markdown("---")
                    st.markdown("**Ver detalhes de uma dessas votações:**")
                    mapa_div = dict(zip(df_div_filtrada['label_curto'], df_div_filtrada['id_votacao_str']))
                    vot_div_sel_label = st.selectbox("Selecione a votação:", list(mapa_div.keys()), key='sb_divididas')
                    vot_div_sel_id = mapa_div[vot_div_sel_label]

                    df_det = df_votos_contagem[
                        (df_votos_contagem['id_votacao_str'] == vot_div_sel_id) &
                        (df_votos_contagem['sgPartido'].isin(partidos_no_filtro))
                    ].copy()

                    if not df_det.empty:
                        fig_det = px.bar(
                            df_det,
                            x='sgPartido', y='perc', color='voto',
                            color_discrete_map={'Sim': '#2ecc71', 'Não': '#e74c3c'},
                            barmode='stack',
                            labels={'perc': '% de Votos', 'sgPartido': 'Partido', 'voto': 'Voto'},
                            template=template_grafico,
                            text_auto='.0f'
                        )
                        fig_det.update_layout(yaxis_title='% de Votos', legend_title='Voto', xaxis={'categoryorder': 'total descending'})
                        st.plotly_chart(fig_det, width='stretch')

    st.divider()

def renderizar_p11():
    # --- 8. P11 (BARRA DE ABAS - RANKING DE PARTIDOS) ---
    st.header("📊 P11 — Ranking de Partidos (Atividade e Gastos)")

    df_partidos_P11 = df_filtrado.groupby('sgPartido').agg({'vlrLiquido': 'sum'}).reset_index()
    df_partidos_P11['perc_frequencia'] = df_partidos_P11['sgPartido'].map(df_freq).fillna(0)

    df_prod_partido = pd.merge(df_autores, df_link_partido_filtrado, on='id_oficial', how='inner')
    contagem_prod_partido = df_prod_partido.groupby('sgPartido').size()
    df_partidos_P11['qtd_proposicoes'] = df_partidos_P11['sgPartido'].map(contagem_prod_partido).fillna(0)

    tab_a, tab_b, tab_c, tab_d = st.tabs(["a) Frequência (Sessões)", "b) Proposições", "c) Gastos Totais", "d) Nuvem de Temas"])

    with tab_a:
        st.subheader("Média de Sessões Deliberativas Comparecidas por Deputado (por Partido)")
        fig_a = px.bar(
            df_partidos_P11.sort_values('perc_frequencia', ascending=False), 
            x='sgPartido', 
            y='perc_frequencia', 
            color='perc_frequencia', 
            color_continuous_scale='Viridis', 
            template=template_grafico, 
            labels={'perc_frequencia': 'Média de Sessões Comparecidas', 'sgPartido': 'Partido'} # <-- Adicionado aqui
        )
        st.plotly_chart(fig_a, width='stretch')

    with tab_b:
        st.subheader("Total de Proposições Legislativas por Partido")
        fig_b = px.bar(
            df_partidos_P11.sort_values('qtd_proposicoes', ascending=False), 
            x='sgPartido', 
            y='qtd_proposicoes', 
            color='qtd_proposicoes', 
            color_continuous_scale='Greens', 
            template=template_grafico, 
            labels={'qtd_proposicoes': 'Número de Proposições', 'sgPartido': 'Partido'} # <-- Adicionado aqui
        )
        st.plotly_chart(fig_b, width='stretch')

    with tab_c:
        st.subheader("Gasto Total Acumulado (Cota Parlamentar) por Partido")
        fig_c = px.bar(
            df_partidos_P11.sort_values('vlrLiquido', ascending=False), 
            x='sgPartido', 
            y='vlrLiquido', 
            color='vlrLiquido', 
            color_continuous_scale='Reds', 
            template=template_grafico, 
            labels={'vlrLiquido': 'Total Gasto (R$)', 'sgPartido': 'Partido'} # <-- Adicionado aqui
        )
        st.plotly_chart(fig_c, width='stretch')

    with tab_d:
        st.subheader("Nuvem de Temas mais Tratados pelo Partido")
        partidos_nuvem = sorted(df_partidos_P11['sgPartido'].unique())
        
        if partidos_nuvem:
            partido_n = st.selectbox("Selecione o Partido para gerar a Nuvem de Palavras:", partidos_nuvem, key='sb_nuvem')
            ids_dep_partido = df_link_partido_filtrado[df_link_partido_filtrado['sgPartido'] == partido_n]['id_oficial'].astype(str).str.strip().unique()
            
            # Filtra as palavras chave dos deputados desse partido
            palavras_partido = df_temas[df_temas['id_oficial'].isin(ids_dep_partido)]['palavras_chave'].dropna()
            
            if len(palavras_partido) > 0:
                freq_partido = obter_frequencia_palavras_chave(palavras_partido)
                
                if not freq_partido.empty:
                    col_nuvem_p, col_tabela_p = st.columns([0.7, 0.3])
                    
                    with col_nuvem_p:
                        wc = WordCloud(
                            width=900, height=500, background_color='white', 
                            colormap='tab10', max_words=80
                        ).generate_from_frequencies(freq_partido.to_dict())
                        
                        fig_d, ax = plt.subplots(figsize=(12, 6))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        plt.tight_layout(pad=0)
                        st.pyplot(fig_d)
                        plt.close(fig_d)
                    
                    with col_tabela_p:
                        df_freq_partido = pd.DataFrame({'Palavra-chave': freq_partido.index, 'Frequência': freq_partido.values})
                        st.dataframe(df_freq_partido, hide_index=True, height=450, use_container_width=True)
                else:
                    st.warning("Não foi possível gerar a nuvem para este partido (palavras insuficientes).")
            else:
                st.warning("Nenhuma palavra-chave disponível nas proposições deste partido.")
        else:
            st.info("Nenhum partido disponível para gerar nuvem com os filtros atuais.")
    st.divider()

def renderizar_p12():
    # =============================================================================
    # --- 12. P12 — DEPUTADO × FORNECEDOR ---
    # =============================================================================
    st.header("🏢 P12 — Deputado × Fornecedor")
    st.markdown(
        "Analisa com quais fornecedores cada deputado mais gastou a cota parlamentar, "
        "e quais fornecedores mais recebem recursos parlamentares no geral."
    )

    df_fornecedores = carregar_fornecedores()
    ids_no_filtro = df_filtrado['id_oficial'].unique()

    # Aplica filtro global
    df_forn_filtrado = df_fornecedores[df_fornecedores['id_oficial'].isin(ids_no_filtro)].copy()

    if df_forn_filtrado.empty:
        st.info("Sem dados de fornecedores para os filtros atuais.")
    else:
        tab_p12_geral, tab_p12_dep, tab_p12_forn = st.tabs([
            "🌎 Top Fornecedores Geral", "👤 Por Deputado", "🔍 Por Fornecedor"
        ])

        # -------------------------------------------------------------------
        # ABA 1: VISÃO GERAL DE FORNECEDORES
        # -------------------------------------------------------------------
        with tab_p12_geral:
            st.subheader("Fornecedores que mais receberam recursos parlamentares")

            col_n12, _ = st.columns([0.3, 0.7])
            with col_n12:
                n_forn = st.slider("Quantos fornecedores exibir?", 5, 50, 20, key='sl_forn_geral')

            top_forn = (
                df_forn_filtrado
                .groupby('fornecedor')
                .agg(
                    total_recebido=('vlrLiquido', 'sum'),
                    n_deputados=('id_oficial', 'nunique'),
                    n_transacoes=('vlrLiquido', 'count')
                )
                .reset_index()
                .sort_values('total_recebido', ascending=False)
                .head(n_forn)
            )

            fig_p12_g = px.bar(
                top_forn,
                x='total_recebido', y='fornecedor',
                orientation='h',
                color='total_recebido', color_continuous_scale='Oranges',
                hover_data={'n_deputados': True, 'n_transacoes': True},
                labels={
                    'total_recebido': 'Total Recebido (R$)',
                    'fornecedor': 'Fornecedor',
                    'n_deputados': 'Nº de Deputados Clientes',
                    'n_transacoes': 'Nº de Transações'
                },
                template=template_grafico,
                text_auto=',.0f'
            )
            fig_p12_g.update_layout(
                yaxis={'categoryorder': 'total ascending', 'type': 'category'},
                coloraxis_showscale=False,
                height=max(400, 150 + (n_forn * 35)),
                bargap=0.15
            )
            fig_p12_g.update_traces(textposition='outside', cliponaxis=False)
            st.plotly_chart(fig_p12_g, width='stretch')

        # -------------------------------------------------------------------
        # ABA 2: VISÃO POR DEPUTADO
        # -------------------------------------------------------------------
        with tab_p12_dep:
            st.subheader("Com quais fornecedores um deputado mais gastou?")

            lista_deps_p12 = sorted(df_forn_filtrado['txNomeParlamentar'].dropna().unique())
            dep_p12 = st.selectbox("Selecione o deputado:", lista_deps_p12, key='sb_dep_p12')

            col_n12b, _ = st.columns([0.3, 0.7])
            with col_n12b:
                n_forn_dep = st.slider("Quantos fornecedores exibir?", 5, 30, 15, key='sl_forn_dep')

            df_dep_selecionado = df_forn_filtrado[df_forn_filtrado['txNomeParlamentar'] == dep_p12]

            forn_dep = (
                df_dep_selecionado
                .groupby(['fornecedor', 'categoria'])
                .agg(total=('vlrLiquido', 'sum'), transacoes=('vlrLiquido', 'count'))
                .reset_index()
                .sort_values('total', ascending=False)
                .head(n_forn_dep)
            )

            if forn_dep.empty:
                st.info(f"Sem dados de fornecedores para {dep_p12}.")
            else:
                total_dep = df_dep_selecionado['vlrLiquido'].sum()
                st.info(f"Total gasto por **{dep_p12}**: R$ {total_dep:,.2f} | Top {n_forn_dep} fornecedores abaixo.")

                fig_p12_dep = px.bar(
                    forn_dep,
                    x='total', y='fornecedor',
                    orientation='h',
                    color='total', color_continuous_scale='Blues', 
                    hover_data={'transacoes': True, 'categoria': True}, 
                    labels={
                        'total': 'Total Gasto (R$)',
                        'fornecedor': 'Fornecedor',
                        'categoria': 'Categoria',
                        'transacoes': 'Nº Transações'
                    },
                    template=template_grafico,
                    text_auto=',.0f'
                )
                fig_p12_dep.update_layout(
                    yaxis={'categoryorder': 'total ascending', 'type': 'category'},
                    coloraxis_showscale=False, 
                    height=max(400, 150 + (n_forn_dep * 35)),
                    bargap=0.15
                )
                fig_p12_dep.update_traces(textposition='outside', cliponaxis=False)
                st.plotly_chart(fig_p12_dep, width='stretch')

                with st.expander("Ver tabela completa de fornecedores"):
                    df_tab_forn = forn_dep.copy()
                    df_tab_forn.columns = ['Fornecedor', 'Categoria', 'Total (R$)', 'Transações']
                    df_tab_forn['Total (R$)'] = df_tab_forn['Total (R$)'].map('R$ {:,.2f}'.format)
                    st.dataframe(df_tab_forn, hide_index=True, use_container_width=True)

        # -------------------------------------------------------------------
        # ABA 3: VISÃO POR FORNECEDOR
        # -------------------------------------------------------------------
        with tab_p12_forn:
            st.subheader("Quais deputados mais pagaram a um fornecedor específico?")

            top500_forn = (
                df_forn_filtrado
                .groupby('fornecedor')['vlrLiquido']
                .sum()
                .sort_values(ascending=False)
                .head(500)
                .index.tolist()
            )

            forn_selecionado = st.selectbox(
                "Selecione o fornecedor (top 500 por volume):",
                top500_forn,
                key='sb_forn_p12'
            )

            col_n12c, _ = st.columns([0.3, 0.7])
            with col_n12c:
                n_deps_forn = st.slider("Quantos deputados exibir?", 5, 30, 15, key='sl_deps_forn')

            df_forn_sel = df_forn_filtrado[df_forn_filtrado['fornecedor'] == forn_selecionado]

            deps_do_forn = (
                df_forn_sel
                .groupby(['txNomeParlamentar', 'sgPartido', 'sgUF'])
                .agg(total=('vlrLiquido', 'sum'), transacoes=('vlrLiquido', 'count'))
                .reset_index()
                .sort_values('total', ascending=False)
                .head(n_deps_forn)
            )

            if deps_do_forn.empty:
                st.info("Sem dados para o fornecedor selecionado.")
            else:
                total_forn = df_forn_sel['vlrLiquido'].sum()
                n_deps_unique = df_forn_sel['id_oficial'].nunique()
                st.info(
                    f"**{forn_selecionado}** recebeu **R$ {total_forn:,.2f}** "
                    f"de **{n_deps_unique}** deputado(s) diferentes."
                )

                fig_p12_forn = px.bar(
                    deps_do_forn,
                    x='total', y='txNomeParlamentar',
                    orientation='h',
                    color='total', color_continuous_scale='Purples', 
                    hover_data={'sgUF': True, 'transacoes': True, 'sgPartido': True}, 
                    labels={
                        'total': 'Total Pago (R$)',
                        'txNomeParlamentar': 'Deputado',
                        'sgPartido': 'Partido',
                        'transacoes': 'Nº Transações'
                    },
                    template=template_grafico,
                    text_auto=',.0f'
                )
                fig_p12_forn.update_layout(
                    yaxis={'categoryorder': 'total ascending', 'type': 'category'},
                    coloraxis_showscale=False, 
                    height=max(400, 150 + (n_deps_forn * 35)),
                    bargap=0.15
                )
                fig_p12_forn.update_traces(textposition='outside', cliponaxis=False)
                st.plotly_chart(fig_p12_forn, width='stretch')

def renderizar_p13():
    # --- 9. P13 (TIPOS DE DESPESA) ---
    st.header("📑 P13 — Tipos de Despesa (Cota Parlamentar)")
    col_vazia_l, col_conteudo_p13, col_vazia_r = st.columns([0.15, 0.7, 0.15])

    with col_conteudo_p13:
        tab_global, tab_individual = st.tabs(["🌎 Visão Geral (Todos no Filtro)", "👤 Visão por Deputado"])
        
        with tab_global:
            cat_data = df_filtrado.groupby('txtDescricao')['vlrLiquido'].sum().reset_index()
            if not cat_data.empty:
                fig_global = px.pie(cat_data, values='vlrLiquido', names='txtDescricao', hole=0.4, template=template_grafico, title="Distribuição de Gastos por Categoria")
                fig_global.update_layout(showlegend=True) 
                st.plotly_chart(fig_global, width='stretch')
            else:
                st.info("Nenhum dado de despesa encontrado para os filtros atuais.")

        with tab_individual:
            lista_nomes_P13 = sorted(df_filtrado['txNomeParlamentar'].dropna().unique())
            if len(lista_nomes_P13) > 0:
                dep_escolhido = st.selectbox("Selecione um deputado para análise individual de gastos:", lista_nomes_P13, key='sb_dep_p13')
                df_ind = df_filtrado[df_filtrado['txNomeParlamentar'] == dep_escolhido]
                cat_ind = df_ind.groupby('txtDescricao')['vlrLiquido'].sum().reset_index()
                
                if not cat_ind.empty:
                    st.info(f"Total gasto acumulado (2023-2026) por {dep_escolhido}: **R$ {df_ind['vlrLiquido'].sum():,.2f}**")
                    fig_ind = px.pie(cat_ind, values='vlrLiquido', names='txtDescricao', hole=0.4, color_discrete_sequence=px.colors.qualitative.Safe, title=f"Distribuição de Gastos de {dep_escolhido}")
                    st.plotly_chart(fig_ind, width='stretch')
                else:
                    st.warning(f"O deputado {dep_escolhido} está no filtro, mas não possui registros de despesas.")
            else:
                st.warning("Nenhum deputado encontrado com os filtros atuais.")

    st.divider()
       

# ==========================================
# CSS PRA ESTILIZAR
# ==========================================
st.markdown("""
    <style>
        /* Força a barra de abas a ocupar 100% da largura da tela */
        div[data-baseweb="tab-list"] {
            display: flex !important;
            width: 100% !important;
        }
        
        /* Ajusta cada aba individualmente para crescer e centralizar */
        button[data-baseweb="tab"] {
            flex: 1 !important; 
            justify-content: center !important; 
        }
        
        /* ATACA DIRETAMENTE O TEXTO DENTRO DA ABA (Aumenta a fonte e deixa Bold) */
        button[data-baseweb="tab"] p,
        button[data-baseweb="tab"] span {
            font-size: 20px !important; /* Aumentado um pouco mais */
            font-weight: bold !important; /* Deixa BEM destacado */
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🌟 NAVEGAÇÃO DO DASHBOARD
# ==========================================
st.subheader("Navegação do Dashboard")

# 1. Cria as abas e guarda nas variáveis
tab_gastos, tab_atuacao, tab_partidos, tab_perfil = st.tabs([
    "💰 Gastos & Cota", 
    "📜 Atuação Legislativa", 
    "🏛️ Dinâmica Partidária", 
    "🎓 Perfil Demográfico"
])


# 2. Preenche a Aba 1
with tab_gastos:
    renderizar_p1()
    renderizar_p12()
    renderizar_p13()

# 3. Preenche a Aba 2
with tab_atuacao:
    renderizar_p2()
    renderizar_p3()
    renderizar_p8()

# 4. Preenche a Aba 3
with tab_partidos:
    renderizar_p10()
    renderizar_p11()

# 5. Preenche a Aba 4
with tab_perfil:
    renderizar_p4()
    renderizar_p6()