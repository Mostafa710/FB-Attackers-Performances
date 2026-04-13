import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib

# ── Data & Model ───────────────────────────────────────────────────────────────
df = pd.read_csv('./datasets/full_dataset.csv')
model = joblib.load('./Model/lr_model.joblib')

MODEL_FEATURES = [
    'age', 'Matches Played', 'Avg Mins per Match', 'Assists',
    'Penalty Kicks Made', 'Assists p 90', 'Possessions lost',
    'Total Shots', '% Shots on target', 'Shots p 90',
    '% Aerial Duels won', 'Shot creating actions p 90'
]
FEATURE_LABELS = {
    'age':                         'Age (years)',
    'Matches Played':              'Matches Played',
    'Avg Mins per Match':          'Avg Minutes per Match',
    'Assists':                     'Total Assists',
    'Penalty Kicks Made':          'Penalty Kicks Made',
    'Assists p 90':                'Assists per 90 min',
    'Possessions lost':            'Possessions Lost',
    'Total Shots':                 'Total Shots',
    '% Shots on target':           '% Shots on Target',
    'Shots p 90':                  'Shots per 90 min',
    '% Aerial Duels won':          '% Aerial Duels Won',
    'Shot creating actions p 90':  'Shot-Creating Actions / 90',
}
FEATURE_DEFAULTS = {
    'age': 25, 'Matches Played': 25, 'Avg Mins per Match': 70,
    'Assists': 5, 'Penalty Kicks Made': 2, 'Assists p 90': 0.20,
    'Possessions lost': 30, 'Total Shots': 60, '% Shots on target': 40.0,
    'Shots p 90': 2.5, '% Aerial Duels won': 45.0,
    'Shot creating actions p 90': 2.5,
}
FEATURE_RANGES = {
    'age':                         (15,  45,   1),
    'Matches Played':              (1,   38,   1),
    'Avg Mins per Match':          (1,   90,   1),
    'Assists':                     (0,   30,   1),
    'Penalty Kicks Made':          (0,   15,   1),
    'Assists p 90':                (0.0, 2.0,  0.01),
    'Possessions lost':            (0,   150,  1),
    'Total Shots':                 (0,   200,  1),
    '% Shots on target':           (0.0, 100.0, 0.5),
    'Shots p 90':                  (0.0, 10.0, 0.1),
    '% Aerial Duels won':          (0.0, 100.0, 0.5),
    'Shot creating actions p 90':  (0.0, 10.0, 0.1),
}

# ── Palette ────────────────────────────────────────────────────────────────────
BG, CARD, BORDER = '#0f1117', '#1a1d27', '#2a2d3e'
A1, A2, A3, A4   = '#6c63ff', '#ff6584', '#43e97b', '#f7971e'
TEXT, MUTED       = '#e2e8f0', '#94a3b8'

LEAGUE_COLORS = {
    'Premier League': '#ef4444', 'La Liga': '#3b82f6',
    'Serie A': '#22c55e', 'Bundesliga': '#a855f7', 'Ligue 1': '#f59e0b',
}
POS_COLORS = {'FW':'#ef4444','MF':'#22c55e','DF':'#3b82f6','GK':'#eab308'}

CARD_S = {'backgroundColor':CARD,'border':f'1px solid {BORDER}',
          'borderRadius':'12px','padding':'20px','marginBottom':'16px'}
TAB_S  = {'backgroundColor':CARD,'color':MUTED,'border':'none',
          'borderRadius':'8px 8px 0 0','padding':'10px 20px','fontWeight':'600','fontSize':'13px'}
TAB_SEL = {**TAB_S, 'color':A1,'borderBottom':f'2px solid {A1}','backgroundColor':BG}

# ── Helpers ────────────────────────────────────────────────────────────────────
def hex_rgba(h, a=0.2):
    h = h.lstrip('#')
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{a})'

def base_layout(fig, h=340):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT, family='Inter, sans-serif', size=11),
        margin=dict(t=25,l=10,r=10,b=10), height=h,
        legend=dict(bgcolor='rgba(0,0,0,0)',bordercolor=BORDER,borderwidth=1,font_size=10),
        xaxis=dict(gridcolor=BORDER,linecolor=BORDER,zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER,linecolor=BORDER,zerolinecolor=BORDER),
    )
    return fig

def stitle(text):
    return html.Div(text, style={'color':TEXT,'fontSize':'13px','fontWeight':'600',
                                  'letterSpacing':'1px','textTransform':'uppercase',
                                  'marginBottom':'14px','paddingBottom':'8px',
                                  'borderBottom':f'1px solid {BORDER}'})

def stat_pill(val, lbl, color):
    return html.Div([
        html.Div(str(val), style={'fontSize':'26px','fontWeight':'700','color':color,'lineHeight':'1'}),
        html.Div(lbl, style={'fontSize':'10px','color':MUTED,'marginTop':'4px'}),
    ], style={'backgroundColor':CARD,'border':f'1px solid {BORDER}','borderRadius':'10px',
              'padding':'14px 18px','textAlign':'center','flex':'1'})

def sid(feat): return 'sl-'+feat.replace(' ','_').replace('%','pct').replace('/','_')
def vid(feat): return 'vl-'+feat.replace(' ','_').replace('%','pct').replace('/','_')

# ── Charts ─────────────────────────────────────────────────────────────────────
def fig_age_hist(pos='FW'):
    sub = df[df['pos']==pos]['age']
    color = POS_COLORS.get(pos, A1)
    fig = go.Figure(go.Histogram(x=sub, nbinsx=24,
        marker=dict(color=color, opacity=0.85, line=dict(color=BG,width=1)),
        hovertemplate='Age: <b>%{x}</b><br>Count: <b>%{y}</b><extra></extra>'))
    mu = sub.mean()
    fig.add_vline(x=mu, line_dash='dash', line_color='white', line_width=1.5,
                  annotation_text=f'μ={mu:.1f}',
                  annotation_font=dict(size=10, color='white'), annotation_position='top right')
    fig.update_xaxes(title_text='Age', title_font_size=11)
    fig.update_yaxes(title_text='Count', title_font_size=11)
    return base_layout(fig, 290)

def fig_pos_donut():
    counts = df['pos'].value_counts().reset_index()
    counts.columns = ['pos','n']
    fig = go.Figure(go.Pie(labels=counts['pos'], values=counts['n'], hole=0.62,
        marker=dict(colors=[POS_COLORS[p] for p in counts['pos']], line=dict(color=BG,width=2)),
        textinfo='label+percent', textfont=dict(size=11, color='white'),
        hovertemplate='<b>%{label}</b>  %{value:,} players  (%{percent})<extra></extra>'))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT),
                      margin=dict(t=5,b=5,l=5,r=5), height=265, showlegend=False)
    return fig

def fig_age_box():
    fig = go.Figure()
    for pos, color in POS_COLORS.items():
        sub = df[df['pos']==pos]['age']
        fig.add_trace(go.Box(y=sub, name=pos,
            marker=dict(color=color,size=3), line=dict(color=color),
            fillcolor=hex_rgba(color,0.18),
            hovertemplate=f'<b>{pos}</b><br>Age: %{{y}}<extra></extra>',
            boxmean=True))
    fig.update_yaxes(title_text='Age', title_font_size=11)
    return base_layout(fig, 265)

def fig_nat_map():
    ISO3 = {
        "Netherlands":                   "NLD",
        "England":                       "GBR",
        "Italy":                         "ITA",
        "Tunisia":                       "TUN",
        "Mauritania":                    "MRT",
        "Algeria":                       "DZA",
        "Argentina":                     "ARG",
        "Albania":                       "ALB",
        "Ghana":                         "GHA",
        "Scotland":                      "GBR",
        "Spain":                         "ESP",
        "Benin":                         "BEN",
        "Germany":                       "DEU",
        "France":                        "FRA",
        "Iraq":                          "IRQ",
        "Democratic Republic of the Congo": "COD",
        "Uruguay":                       "URY",
        "Ivory Coast":                   "CIV",
        "Brazil":                        "BRA",
        "Morocco":                       "MAR",
        "Switzerland":                   "CHE",
        "Nigeria":                       "NGA",
        "Austria":                       "AUT",
        "Chile":                         "CHL",
        "Belgium":                       "BEL",
        "Cameroon":                      "CMR",
        "Wales":                         "GBR",
        "Paraguay":                      "PRY",
        "Indonesia":                     "IDN",
        "Senegal":                       "SEN",
        "Denmark":                       "DNK",
        "Romania":                       "ROU",
        "Jamaica":                       "JAM",
        "Portugal":                      "PRT",
        "Canada":                        "CAN",
        "Sweden":                        "SWE",
        "Ireland":                       "IRL",
        "Japan":                         "JPN",
        "Gabon":                         "GAB",
        "Republic of the Congo":         "COG",
        "French Guiana":                 "GUF",
        "Colombia":                      "COL",
        "Croatia":                       "HRV",
        "Bosnia and Herzegovina":        "BIH",
        "Czech Republic":                "CZE",
        "North Macedonia":               "MKD",
        "Gambia":                        "GMB",
        "Serbia":                        "SRB",
        "Angola":                        "AGO",
        "Haiti":                         "HTI",
        "Guadeloupe":                    "GLP",
        "Cape Verde":                    "CPV",
        "Togo":                          "TGO",
        "Poland":                        "POL",
        "Slovenia":                      "SVN",
        "Libya":                         "LBY",
        "Slovakia":                      "SVK",
        "Curaçao":                       "CUW",
        "Burundi":                       "BDI",
        "Iceland":                       "ISL",
        "Mali":                          "MLI",
        "Costa Rica":                    "CRI",
        "United States":                 "USA",
        "Northern Ireland":              "GBR",
        "Liechtenstein":                 "LIE",
        "Martinique":                    "MTQ",
        "Ecuador":                       "ECU",
        "Turkey":                        "TUR",
        "Guinea":                        "GIN",
        "Uganda":                        "UGA",
        "Peru":                          "PER",
        "South Korea":                   "KOR",
        "Russia":                        "RUS",
        "Norway":                        "NOR",
        "Venezuela":                     "VEN",
        "Burkina Faso":                  "BFA",
        "Saudi Arabia":                  "SAU",
        "South Africa":                  "ZAF",
        "Greece":                        "GRC",
        "Egypt":                         "EGY",
        "Mexico":                        "MEX",
        "Dominican Republic":            "DOM",
        "Bulgaria":                      "BGR",
        "Hungary":                       "HUN",
        "Kosovo":                        "XKX",
        "Israel":                        "ISR",
        "Finland":                       "FIN",
        "Philippines":                   "PHL",
        "Guinea-Bissau":                 "GNB",
        "Moldova":                       "MDA",
        "Montenegro":                    "MNE",
        "Sierra Leone":                  "SLE",
        "Laos":                          "LAO",
        "Estonia":                       "EST",
        "Niger":                         "NER",
        "Central African Republic":      "CAF",
        "Ukraine":                       "UKR",
        "North Korea":                   "PRK",
        "Madagascar":                    "MDG",
        "New Caledonia":                 "NCL",
        "Australia":                     "AUS",
        "Honduras":                      "HND",
        "Comoros":                       "COM",
        "Equatorial Guinea":             "GNQ",
        "Mozambique":                    "MOZ",
        "Armenia":                       "ARM",
        "Chad":                          "TCD",
        "Kenya":                         "KEN",
        "Luxembourg":                    "LUX",
        "New Zealand":                   "NZL",
        "Bermuda":                       "BMU",
        "Trinidad and Tobago":           "TTO",
        "Iran":                          "IRN",
        "Cyprus":                        "CYP",
        "China":                         "CHN",
        "Georgia":                       "GEO",
        "Guatemala":                     "GTM",
        "Suriname":                      "SUR",
        "Bolivia":                       "BOL",
        "Cuba":                          "CUB",
        "Zimbabwe":                      "ZWE",
        "Tanzania":                      "TZA",
        "Zambia":                        "ZMB",
        "Saint Kitts and Nevis":         "KNA",
        "Uzbekistan":                    "UZB",
        "Faroe Islands":                 "FRO",
        "Panama":                        "PAN",
        "Grenada":                       "GRD",
        "Lithuania":                     "LTU",
        "Syria":                         "SYR",
        "Jordan":                        "JOR",
        "Malta":                         "MLT",}
    nc = (df.drop_duplicates(subset=['player','nation'])[['nation']]
            .groupby('nation').size().reset_index(name='count'))
    nc['iso3'] = nc['nation'].map(ISO3)
    nc = nc.dropna(subset=['iso3']).groupby('iso3', as_index=False).agg(
        count=('count','sum'), nation=('nation', lambda x: ' / '.join(sorted(x))))
    fig = go.Figure(go.Choropleth(
        locations=nc['iso3'], z=nc['count'], text=nc['nation'],
        colorscale=[[0,'#1a1d27'],[0.3,'#3b3f5c'],[0.65,A1],[1,'#a78bfa']],
        autocolorscale=False,
        marker=dict(line=dict(color=BORDER,width=0.5)),
        colorbar=dict(title=dict(text='Players',font=dict(size=10,color=MUTED)),
                      tickfont=dict(color=MUTED,size=9), bgcolor='rgba(0,0,0,0)',
                      thickness=12, len=0.6),
        hovertemplate='<b>%{text}</b><br>Players: <b>%{z}</b><extra></extra>',
        zmin=0, zmax=nc['count'].quantile(0.97)))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT),
                      margin=dict(t=0,l=0,r=0,b=0), height=295,
                      geo=dict(showframe=False, showcoastlines=True, coastlinecolor=BORDER,
                               showland=True, landcolor='#1a1d27', showocean=True, oceancolor=BG,
                               showlakes=False, showcountries=True, countrycolor=BORDER,
                               projection_type='natural earth', bgcolor='rgba(0,0,0,0)'))
    return fig

def fig_league_radar():
    cols = ['Pass completion %','Goals p 90','Expected Goals','Tackles Won',
            '% Successful take-ons','Progressive Passes','Shot creating actions p 90','% Aerial Duels won']
    lbls = ['Pass %','Goals/90','xG','Tackles','Dribble %','Prog Pass','SCA/90','Aerial %']
    avg  = df.groupby('comp')[cols].mean().reset_index()
    norm = avg.copy()
    for c in cols:
        mn,mx = avg[c].min(), avg[c].max()
        norm[c] = (avg[c]-mn)/(mx-mn+1e-9)
    cl = lbls+[lbls[0]]; cc = cols+[cols[0]]
    fig = go.Figure()
    for _, row in norm.iterrows():
        lg = row['comp']; color = LEAGUE_COLORS.get(lg,'#888')
        raw = avg[avg['comp']==lg].iloc[0]
        hover = [f'<b>{lg}</b><br>{lbls[i]}: <b>{raw[cols[i]]:.2f}</b>' for i in range(len(cols))]+[f'<b>{lg}</b>']
        fig.add_trace(go.Scatterpolar(r=[row[c] for c in cc], theta=cl,
            fill='toself', fillcolor=color, opacity=0.5,
            line=dict(color=color,width=2.2), name=lg,
            text=hover, hovertemplate='%{text}<extra></extra>'))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT,size=10),
        margin=dict(t=25,l=10,r=10,b=10), height=360,
        polar=dict(bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True,range=[0,1],tickvals=[.25,.5,.75,1],
                            ticktext=['25%','50%','75%','100%'],
                            tickfont=dict(size=8,color=MUTED),gridcolor=BORDER,linecolor=BORDER),
            angularaxis=dict(tickfont=dict(size=10,color=TEXT),gridcolor=BORDER,linecolor=BORDER)),
        legend=dict(bgcolor='rgba(0,0,0,0)',font_size=10,orientation='h',x=0.5,xanchor='center',y=-0.18),
        showlegend=True)
    return fig

def fig_xg_trend():
    xg = df.groupby(['comp','season'])['Expected Goals'].mean().reset_index()
    fig = go.Figure()
    for lg, color in LEAGUE_COLORS.items():
        sub = xg[xg['comp']==lg].sort_values('season')
        fig.add_trace(go.Scatter(x=sub['season'],y=sub['Expected Goals'],
            mode='lines+markers', name=lg,
            line=dict(color=color,width=2.5),
            marker=dict(color=color,size=7,line=dict(color=BG,width=1.5)),
            hovertemplate=f'<b>{lg}</b><br>%{{x}}<br>Avg xG: <b>%{{y:.2f}}</b><extra></extra>'))
    fig.update_xaxes(title_text='Season', title_font_size=11)
    fig.update_yaxes(title_text='Avg xG', title_font_size=11)
    return base_layout(fig, 305)

def fig_top_xg():
    fw = df[(df['pos']=='FW') & (df['Avg Mins per Match']*df['Matches Played']>=450)]
    top = fw.groupby('player')['Expected Goals'].mean().nlargest(15).reset_index().sort_values('Expected Goals')
    n = len(top)
    colors = [f'rgba(108,99,255,{0.45+0.55*i/max(n-1,1)})' for i in range(n)]
    fig = go.Figure(go.Bar(x=top['Expected Goals'],y=top['player'],orientation='h',
        marker=dict(color=colors,line=dict(color=BORDER,width=0.5)),
        hovertemplate='<b>%{y}</b><br>Avg xG: <b>%{x:.2f}</b><extra></extra>'))
    fig.update_xaxes(title_text='Avg Expected Goals', title_font_size=11)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT,family='Inter, sans-serif',size=11),
        margin=dict(t=10,l=20,r=20,b=10), height=400,
        yaxis=dict(gridcolor=BORDER,linecolor=BORDER),
        xaxis=dict(gridcolor=BORDER,linecolor=BORDER))
    return fig

def fig_goals_shots():
    fw = df[df['pos']=='FW'].sample(min(700,len(df[df['pos']=='FW'])),random_state=1)
    fig = go.Figure(go.Scatter(x=fw['Total Shots'],y=fw['Goals'],mode='markers',
        marker=dict(color=fw['Expected Goals'],colorscale='Plasma',size=6,opacity=0.65,
                    line=dict(color=BORDER,width=0.3),showscale=True,
                    colorbar=dict(title='xG',thickness=11,len=0.75,
                                  tickfont=dict(size=9,color=MUTED),
                                  title_font=dict(size=10,color=MUTED))),
        hovertemplate='Shots: <b>%{x}</b><br>Goals: <b>%{y}</b><br>xG: <b>%{marker.color:.1f}</b><extra></extra>'))
    fig.update_xaxes(title_text='Total Shots', title_font_size=11)
    fig.update_yaxes(title_text='Goals', title_font_size=11)
    return base_layout(fig, 315)

def fig_xg_actual():
    fw = df[df['pos']=='FW'].sample(min(600,len(df[df['pos']=='FW'])),random_state=42)
    fig = go.Figure()
    for lg, color in LEAGUE_COLORS.items():
        sub = fw[fw['comp']==lg]
        fig.add_trace(go.Scatter(x=sub['Expected Goals'],y=sub['Goals'],mode='markers',name=lg,
            marker=dict(color=color,size=5,opacity=0.6,line=dict(color=BG,width=0.3)),
            hovertemplate=f'<b>{lg}</b><br>xG: %{{x:.1f}}<br>Goals: %{{y}}<extra></extra>'))
    mx = max(fw['Expected Goals'].max(), fw['Goals'].max())
    fig.add_trace(go.Scatter(x=[0,mx],y=[0,mx],mode='lines',
        line=dict(color='white',dash='dash',width=1),showlegend=False,hoverinfo='skip'))
    fig.update_xaxes(title_text='Expected Goals (xG)', title_font_size=11)
    fig.update_yaxes(title_text='Actual Goals', title_font_size=11)
    return base_layout(fig, 315)

def fig_shot_acc():
    fw = df[df['pos']=='FW']
    stats = fw.groupby('comp')['% Shots on target'].mean().reset_index().sort_values('% Shots on target')
    colors = [LEAGUE_COLORS.get(lg,A1) for lg in stats['comp']]
    fig = go.Figure(go.Bar(x=stats['% Shots on target'],y=stats['comp'],orientation='h',
        marker=dict(color=colors,line=dict(color=BORDER,width=0.5)),
        hovertemplate='<b>%{y}</b><br>Shot Acc: <b>%{x:.1f}%</b><extra></extra>'))
    fig.update_xaxes(title_text='Avg % Shots on Target', title_font_size=11)
    return base_layout(fig, 260)

def fig_gauge(value):
    fig = go.Figure(go.Indicator(mode='gauge+number', value=value if value else 0,
        number=dict(font=dict(color=A1,size=30), suffix=' xG'),
        gauge=dict(
            axis=dict(range=[0,30],tickfont=dict(color=MUTED,size=9),tickcolor=BORDER,nticks=7),
            bar=dict(color=A1,thickness=0.35), bgcolor='rgba(0,0,0,0)', borderwidth=0,
            steps=[dict(range=[0,5],   color=hex_rgba('#ef4444',0.15)),
                   dict(range=[5,12],  color=hex_rgba('#eab308',0.15)),
                   dict(range=[12,20], color=hex_rgba('#22c55e',0.15)),
                   dict(range=[20,30], color=hex_rgba(A1,0.15))],
            threshold=dict(line=dict(color='white',width=2),thickness=0.7,value=value or 0))))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT),
                      height=195, margin=dict(t=20,b=20,l=35,r=35))
    return fig

# ── Tab builders ───────────────────────────────────────────────────────────────
def make_overview():
    return html.Div([
        html.Div([stat_pill(f"{len(df):,}","Total Records",A1),
                  stat_pill(f"{df['player'].nunique():,}","Unique Players",A2),
                  stat_pill(f"{df['nation'].nunique()}","Nationalities",A3),
                  stat_pill(f"{df['squad'].nunique()}","Clubs",A4),
                  stat_pill(f"{df['season'].nunique()}","Seasons",'#38bdf8')],
                 style={'display':'flex','gap':'12px','marginBottom':'20px'}),
        html.Div([
            html.Div([stitle('Position Split'), dcc.Graph(figure=fig_pos_donut(), config={'displayModeBar':False})],
                     style={**CARD_S,'flex':'1'}),
            html.Div([
                html.Div([stitle('Age Distribution'),
                          dcc.Dropdown(id='dd-pos',
                              options=[{'label':p,'value':p} for p in ['FW','MF','DF','GK']],
                              value='FW', clearable=False,
                              style={'width':'85px','fontSize':'12px'})],
                         style={'display':'flex','justifyContent':'space-between','alignItems':'flex-start'}),
                dcc.Graph(id='g-age-hist', config={'displayModeBar':False}),
            ], style={**CARD_S,'flex':'2'}),
            html.Div([stitle('Age Range by Position'), dcc.Graph(figure=fig_age_box(), config={'displayModeBar':False})],
                     style={**CARD_S,'flex':'1.2'}),
        ], style={'display':'flex','gap':'16px'}),
        html.Div([stitle('Player Nationality Map'), dcc.Graph(figure=fig_nat_map(), config={'displayModeBar':False})],
                 style=CARD_S),
    ])

def make_leagues():
    return html.Div([
        html.Div([
            html.Div([stitle('League Style Fingerprint'),
                      html.Div('Metrics normalised 0–1  ·  hover for raw values',
                               style={'fontSize':'10px','color':MUTED,'marginTop':'-10px','marginBottom':'12px'}),
                      dcc.Graph(figure=fig_league_radar(), config={'displayModeBar':False})],
                     style={**CARD_S,'flex':'1.1'}),
            html.Div([stitle('Average xG Trend per Season'),
                      dcc.Graph(figure=fig_xg_trend(), config={'displayModeBar':False})],
                     style={**CARD_S,'flex':'2'}),
        ], style={'display':'flex','gap':'16px'}),
        html.Div([stitle('League Averages Summary'), html.Div(id='league-tbl')], style=CARD_S),
    ])

def make_attackers():
    return html.Div([
        html.Div([
            html.Div([stitle('Top 15 xG Leaders  (FW · ≥450 min)'),
                      dcc.Graph(figure=fig_top_xg(), config={'displayModeBar':False})],
                     style={**CARD_S,'flex':'1'}),
            html.Div([stitle('Goals vs Total Shots  (colour = xG)'),
                      dcc.Graph(figure=fig_goals_shots(), config={'displayModeBar':False})],
                     style={**CARD_S,'flex':'1.3'}),
        ], style={'display':'flex','gap':'16px'}),
        html.Div([
            html.Div([stitle('Shot Accuracy by League  (FW)'),
                      dcc.Graph(figure=fig_shot_acc(), config={'displayModeBar':False})],
                     style={**CARD_S,'flex':'1'}),
            html.Div([stitle('xG vs Actual Goals  (FW sample)'),
                      dcc.Graph(figure=fig_xg_actual(), config={'displayModeBar':False})],
                     style={**CARD_S,'flex':'1'}),
        ], style={'display':'flex','gap':'16px'}),
    ])

def make_predictor():
    def slider_row(feat):
        mn, mx, step = FEATURE_RANGES[feat]
        return html.Div([
            html.Div([
                html.Span(FEATURE_LABELS[feat],
                          style={'fontSize':'13px','color':TEXT,'fontWeight':'500'}),
                html.Span(f'  [{mn} – {mx}]', style={'fontSize':'10px','color':MUTED}),
            ], style={'marginBottom':'4px'}),
            html.Div([
                dcc.Slider(id=sid(feat), min=mn, max=mx, step=step,
                           value=FEATURE_DEFAULTS[feat], marks=None,
                           tooltip={'placement':'bottom','always_visible':False}),
                html.Div(id=vid(feat), children=str(FEATURE_DEFAULTS[feat]),
                         style={'fontSize':'12px','color':A1,'fontWeight':'700',
                                'minWidth':'48px','textAlign':'right'}),
            ], style={'display':'flex','alignItems':'center','gap':'12px'}),
        ], style={'marginBottom':'18px'})

    model_info_rows = [
        ('Algorithm', 'Linear Regression'),
        ('Target', 'Expected Goals (xG)'),
        ('Position', 'Forwards (FW)'),
        ('Features', str(len(MODEL_FEATURES))),
        ('Split', '80% train / 20% test'),
    ]

    return html.Div([
        html.Div([
            # ── Left: sliders ──────────────────────────────────────────────────
            html.Div([
                stitle('Player Statistics Input'),
                html.Div('Adjust sliders to match a forward\'s season stats, then click Predict.',
                         style={'fontSize':'11px','color':MUTED,'marginBottom':'20px'}),
                *[slider_row(f) for f in MODEL_FEATURES],
                html.Button('⚡  Predict xG', id='btn-predict',
                    style={'backgroundColor':A1,'color':'white','border':'none',
                           'borderRadius':'8px','padding':'12px 28px','fontSize':'14px',
                           'fontWeight':'700','cursor':'pointer','width':'100%','marginTop':'4px',
                           'boxShadow':f'0 4px 18px {hex_rgba(A1,0.35)}'}),
            ], style={**CARD_S,'flex':'1.3','maxHeight':'820px','overflowY':'auto'}),

            # ── Right: results ─────────────────────────────────────────────────
            html.Div([
                html.Div([
                    stitle('xG Prediction'),
                    html.Div(id='pred-result', children=[
                        html.Div('—', style={'fontSize':'68px','fontWeight':'800',
                                              'color':A1,'textAlign':'center','lineHeight':'1'}),
                        html.Div('Expected Goals', style={'fontSize':'13px','color':MUTED,
                                                           'textAlign':'center','marginTop':'6px'}),
                        html.Div('Click "Predict xG" to generate',
                                 style={'fontSize':'10px','color':MUTED,'textAlign':'center','marginTop':'14px'}),
                    ]),
                ], style=CARD_S),

                html.Div([
                    stitle('xG Gauge'),
                    dcc.Graph(id='g-gauge', figure=fig_gauge(None), config={'displayModeBar':False}),
                ], style=CARD_S),

                html.Div([
                    stitle('Model Information'),
                    *[html.Div([
                        html.Span(k, style={'color':MUTED,'fontSize':'12px'}),
                        html.Span(v, style={'color':TEXT,'fontSize':'12px','fontWeight':'600'}),
                      ], style={'display':'flex','justifyContent':'space-between','marginBottom':'8px'})
                      for k,v in model_info_rows],
                ], style=CARD_S),
            ], style={'flex':'1','display':'flex','flexDirection':'column'}),
        ], style={'display':'flex','gap':'16px','alignItems':'flex-start'}),
    ])

# ── App ────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap',
])

server = app.server

navbar = html.Div([
    html.Div([
        html.Span('⚽', style={'fontSize':'30px','marginRight':'10px'}),
        html.Div([
            html.Div('FB Players Performance',
                     style={'fontSize':'17px','fontWeight':'700','color':TEXT}),
            html.Div('Top 5 European Leagues  ·  2017–2024',
                     style={'fontSize':'12px','color':MUTED}),
        ]),
    ], style={'display':'flex','alignItems':'center'}),
    html.Div([
        html.Span(f"{len(df):,} records", style={
            'backgroundColor':hex_rgba(A1,0.15),'color':A1,
            'border':f'1px solid {hex_rgba(A1,0.3)}','borderRadius':'20px',
            'padding':'4px 12px','fontSize':'11px','fontWeight':'600','marginRight':'8px'}),
        html.Span('5 leagues  ·  7 seasons', style={
            'backgroundColor':hex_rgba(A3,0.15),'color':A3,
            'border':f'1px solid {hex_rgba(A3,0.3)}','borderRadius':'20px',
            'padding':'4px 12px','fontSize':'11px','fontWeight':'600'}),
    ]),
], style={'backgroundColor':CARD,'borderBottom':f'2px solid {A1}',
          'padding':'14px 28px','display':'flex','justifyContent':'space-between',
          'alignItems':'center','position':'sticky','top':'0','zIndex':'100'})

app.layout = html.Div([
    navbar,
    html.Div([
        dcc.Tabs(id='tabs', value='overview', children=[
            dcc.Tab(label='📊  Overview',     value='overview',  style=TAB_S, selected_style=TAB_SEL),
            dcc.Tab(label='🏆  Leagues',      value='leagues',   style=TAB_S, selected_style=TAB_SEL),
            dcc.Tab(label='⚽  Attackers',    value='attackers', style=TAB_S, selected_style=TAB_SEL),
            dcc.Tab(label='🤖  xG Predictor', value='predictor', style=TAB_S, selected_style=TAB_SEL),
        ], style={'backgroundColor':CARD,'borderBottom':f'1px solid {BORDER}','marginBottom':'20px'}),
        html.Div(id='tab-body'),
    ], style={'padding':'20px 28px','maxWidth':'1600px','margin':'0 auto'}),
], style={'backgroundColor':BG,'minHeight':'100vh','fontFamily':'Inter, sans-serif'})

# ── Callbacks ──────────────────────────────────────────────────────────────────
@app.callback(Output('tab-body','children'), Input('tabs','value'))
def render(tab):
    if tab == 'overview':  return make_overview()
    if tab == 'leagues':   return make_leagues()
    if tab == 'attackers': return make_attackers()
    if tab == 'predictor': return make_predictor()
    return make_overview()

@app.callback(Output('g-age-hist','figure'), Input('dd-pos','value'))
def update_age_hist(pos):
    return fig_age_hist(pos)

@app.callback(Output('league-tbl','children'), Input('tabs','value'))
def build_league_tbl(tab):
    if tab != 'leagues': return dash.no_update
    metrics = ['Pass completion %','Expected Goals','Tackles Won',
               '% Successful take-ons','Progressive Passes',
               'Shot creating actions p 90','% Aerial Duels won']
    pretty  = ['Pass %','Avg xG','Tackles Won','Dribble %','Prog Passes','SCA/90','Aerial %']
    tbl = df.groupby('comp')[metrics].mean().round(2).reset_index()
    th = lambda t, right=False: html.Th(t, style={
        'color':MUTED,'fontSize':'11px','fontWeight':'600',
        'padding':'8px 14px','borderBottom':f'1px solid {BORDER}',
        'textAlign':'right' if right else 'left'})
    header = html.Tr([th('League')] + [th(p, right=True) for p in pretty])
    rows = []
    for _, row in tbl.iterrows():
        lg = row['comp']
        color = LEAGUE_COLORS.get(lg, TEXT)
        cells = [html.Td(lg, style={'color':color,'fontWeight':'700','fontSize':'13px',
                                     'padding':'10px 14px','borderBottom':f'1px solid {BORDER}22'})]
        for m in metrics:
            cells.append(html.Td(f'{row[m]:.2f}', style={'color':TEXT,'fontSize':'13px',
                'padding':'10px 14px','textAlign':'right','borderBottom':f'1px solid {BORDER}22'}))
        rows.append(html.Tr(cells))
    return html.Table([html.Thead(header),html.Tbody(rows)],
                      style={'width':'100%','borderCollapse':'collapse'})

# Slider live-value labels
for _feat in MODEL_FEATURES:
    _sid, _vid = sid(_feat), vid(_feat)
    @app.callback(Output(_vid,'children'), Input(_sid,'value'))
    def _show_val(v):
        if v is None: return '—'
        fv = float(v)
        return f'{fv:.2f}' if fv % 1 != 0 else str(int(fv))

# Prediction
@app.callback(
    Output('pred-result','children'),
    Output('g-gauge','figure'),
    Input('btn-predict','n_clicks'),
    [State(sid(f),'value') for f in MODEL_FEATURES],
    prevent_initial_call=True,
)
def predict(n, *vals):
    vals = [v if v is not None else FEATURE_DEFAULTS[f]
            for v, f in zip(vals, MODEL_FEATURES)]
    pred = max(0.0, float(model.predict(pd.DataFrame([vals], columns=MODEL_FEATURES))[0]))

    fw_xg = df[df['pos']=='FW']['Expected Goals']
    pct   = float((fw_xg < pred).mean() * 100)
    top   = 100 - pct

    if pred >= 15:   lc = A1
    elif pred >= 8:  lc = A3
    elif pred >= 4:  lc = A4
    else:            lc = A2

    result = html.Div([
        html.Div(f'{pred:.2f}', style={'fontSize':'68px','fontWeight':'800','color':lc,
                                        'textAlign':'center','lineHeight':'1','letterSpacing':'-2px'}),
        html.Div('Expected Goals (xG)', style={'fontSize':'13px','color':MUTED,
                                                'textAlign':'center','marginTop':'6px'}),
        html.Div(f'Top {top:.0f}% of forwards in dataset',
                 style={'fontSize':'12px','color':lc,'textAlign':'center',
                        'marginTop':'12px','fontWeight':'600'}),
        html.Div([
            html.Div(style={'height':'5px','borderRadius':'3px','position':'relative',
                            'background':f'linear-gradient(to right, {A2}, {A4}, {A3}, {A1})',
                            'marginTop':'20px'},
                children=[html.Div(style={'position':'absolute','top':'-5px',
                    'left':f'{min(pct,96):.0f}%','width':'14px','height':'14px',
                    'backgroundColor':'white','borderRadius':'50%',
                    'border':f'2px solid {lc}','transform':'translateX(-50%)'})]),
            html.Div([html.Span('Low',style={'fontSize':'10px','color':MUTED}),
                      html.Span('Avg',style={'fontSize':'10px','color':MUTED}),
                      html.Span('Elite',style={'fontSize':'10px','color':MUTED})],
                     style={'display':'flex','justifyContent':'space-between','marginTop':'4px'}),
        ]),
    ])
    return result, fig_gauge(pred)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)
