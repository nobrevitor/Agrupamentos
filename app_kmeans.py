import pandas as pd
from pandas.api.types import is_object_dtype
from pandas.api.types import is_bool_dtype

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA

def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(page_title = 'RFV', \
        layout="wide",
        initial_sidebar_state='expanded'
    )

    # Título principal da aplicação
    st.write("""# K-Means
             
             Modelagem de agrupamento de dados
    """)
    
    @st.cache_data
    def distribuicao (df: pd.DataFrame, var: str, limite_discretas=40):
        '''
        Recebe um DataFrame e uma variável:
        
        Devolve um histplot com um boxplot junto, diferenciando variável quantítativa contínua e discreta;
        
        Devolve um countplot para as variáveis qualitativas ou booleanas;
        '''
        
        if is_object_dtype(df[var]) or is_bool_dtype(df[var]):
            
            fig = plt.figure(figsize=(10,6))
            
            ordem = df[var].value_counts().index
            ax = sns.countplot(df, x=var, hue=var, palette='crest', order=ordem,legend=False)
            
            for container in ax.containers:
                ax.bar_label(container, fontsize=10)
            
            ax.set(title = f'Distribuição da Quantidade por {var.capitalize()}', ylabel = 'Quantidade', xlabel = var.capitalize())
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)

            st.pyplot(fig)
            
        else:
            
            unicos = df[var].nunique()
            if unicos <= limite_discretas:
                tipo = 'discreta'
            else:
                tipo = 'contínua'
            
            fig = plt.figure(figsize=(10, 6))

            if tipo == 'contínua':       
                gs = fig.add_gridspec(2, 1, height_ratios=(1, 4), hspace=0.05)

                ax_box = fig.add_subplot(gs[0])
                ax_hist = fig.add_subplot(gs[1], sharex=ax_box)

                sns.boxplot(x=df[var], ax=ax_box, color= sns.color_palette('crest')[3])
                ax_box.set(title=f'Distribuição da variável contínua: {var}')
                ax_box.set_ylabel('BoxPlot', rotation=0, labelpad=20)
                ax_box.grid(True, linestyle='--', alpha=0.3)

                sns.histplot(df[var], bins=30, ax=ax_hist, color=sns.color_palette('crest')[3])
                ax_hist.set(ylabel='Contagem')
                ax_hist.grid(True, linestyle='--', alpha=0.3)

                ax_box.tick_params(axis='x', labelbottom=True)
                
            else:
                gs = fig.add_gridspec(2, 1, height_ratios=(1, 4), hspace=0.05)

                ax_box = fig.add_subplot(gs[0])
                ax_count = fig.add_subplot(gs[1], sharex=ax_box)

                sns.boxplot(x=df[var], ax=ax_box, color= sns.color_palette('crest')[0])
                ax_box.set(title=f'Distribuição da variável discreta: {var}')
                ax_box.set_ylabel('BoxPlot', rotation=0, labelpad=20)
                ax_box.grid(True, linestyle='--', alpha=0.3)


                sns.countplot(x=df[var], ax=ax_count, color= sns.color_palette('crest')[0], edgecolor='black')
                ax_count.set(ylabel='Contagem')
                ax_count.grid(True, linestyle='--', alpha=0.3)


                ax_box.tick_params(axis='x', labelbottom=True)
                
            st.pyplot(fig)
            
    @st.cache_data        
    def tratamento(df, vars):
        pt = PowerTransformer(method='yeo-johnson')
        yeo = df.copy()
        yeo[vars] = pt.fit_transform(yeo[vars])
        
        n = len(vars)
        fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, 4 * n))
        
        fig.suptitle('Distribuições Yeo-Johnson vs Distribuições Normais', fontsize=16)


        for i, var in enumerate(vars):
            
            sns.histplot(yeo[var], bins=50, ax=axes[i, 0], color= sns.color_palette('crest')[4])
            axes[i, 0].set(title= f'{var} Yeo-Johnson', xlabel='', ylabel='Contagem')
            axes[i, 0].grid(True, linestyle='--', alpha=0.3)
            
            sns.histplot(df[var], bins=50, ax=axes[i, 1], color= sns.color_palette('crest')[0])
            axes[i, 1].set(title= f'{var} Original', xlabel='', ylabel='Contagem')
            axes[i, 1].grid(True, linestyle='--', alpha=0.3)


        plt.tight_layout(rect=[0, 0, 1, 0.97])
        st.pyplot(fig)
            
    @st.cache_data
    def biplot(score,coeff, y, labels=None):
        
        xs = score[:,0]
        ys = score[:,1]
        n = coeff.shape[0]
        scalex = 2/(xs.max() - xs.min())
        scaley = 2/(ys.max() - ys.min())
        
        fig, ax = plt.subplots(figsize=(10, 10))
    #     scatter = ax.scatter(xs * scalex,ys * scaley, c = y)
        sns.kdeplot(x = xs * scalex, y = ys * scaley, hue=y, ax=ax, fill=True, alpha=.6, palette='crest')
    #     ax.legend(*scatter.legend_elements())
        
        for i in range(n):
            ax.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5, 
                    length_includes_head=True, head_width=0.04, head_length=0.04)
            if labels is None:
                ax.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'k', ha = 'center', va = 'center')
            else:
                ax.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'k', ha = 'center', va = 'center')
        ax.set_xlim(-1.2,1.2)
        ax.set_ylim(-1.2,1.2)
        ax.set_xlabel("PC{0}, {1:.1%} da variância explicada".format(1, pca.explained_variance_ratio_[0]))
        ax.set_ylabel("PC{0}, {1:.1%} da variância explicada".format(1, pca.explained_variance_ratio_[1]))
        ax.grid()
        st.pyplot(fig)
        
    @st.cache_data    
    def silhueta(df):
        silhuetas = []
        max_clusters = 15

        dados_cluster = df.copy()

        for n_clusters in range(2, max_clusters + 1):

            km = KMeans(n_clusters=n_clusters, random_state=42).fit(dados_cluster)

            silhuetas.append(silhouette_score(dados_cluster, km.labels_))
            
            df[f'grupos_{n_clusters}'] = pd.Categorical(['grupo_' + str(g) for g in km.labels_])
            
        df_silhueta = pd.DataFrame({'n_clusters': list(range(2, max_clusters+1)), 'silhueta_media': silhuetas})

        fig = plt.figure(figsize=(10,6))

        ax = sns.lineplot(df_silhueta, x = 'n_clusters', y = 'silhueta_media', marker='o')
        ax.set(title='Número de cluster por silhueta média', xlabel='Número de clusters', ylabel='Silhueta Média')
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
    @st.cache_data
    def proporcao(df):
        proporcao_df = round(pd.crosstab(df['Grupos'], df['Revenue'], normalize='index') * 100, 2)
        proporcao_df.rename(columns={True: 'Verdadeiro', False: 'Falso'}, inplace=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Barras empilhadas
        ax.bar(proporcao_df.index, proporcao_df['Falso'], label='Falso', color='#D9544D')
        ax.bar(proporcao_df.index, proporcao_df['Verdadeiro'], bottom=proporcao_df['Falso'], label='Verdadeiro', color='#5FB760')

        # Títulos e rótulos
        ax.set_title('Distribuição de Compras por Grupo de Usuário', fontsize=14)
        ax.set_ylabel('Porcentagem (%)', fontsize=12)
        ax.set_xlabel('Grupos', fontsize=12)
        ax.set_xticks(range(len(proporcao_df.index)))
        ax.set_xticklabels(proporcao_df.index, rotation=0)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(title='Compra Realizada', title_fontsize=11, loc='lower left')

        # Rótulos nas barras
        for i, grupo in enumerate(proporcao_df.index):
            falso = proporcao_df.loc[grupo, 'Falso']
            verdadeiro = proporcao_df.loc[grupo, 'Verdadeiro']

            ax.text(i, falso / 2, f'{falso:.1f}%', ha='center', va='center', color='white', fontsize=10)
            ax.text(i, falso + verdadeiro / 2, f'{verdadeiro:.1f}%', ha='center', va='center', color='white', fontsize=10)

        st.pyplot(fig)
            
    st.markdown("---")
    
    # Apresenta a imagem na barra lateral da aplicação
    # image = Image.open("Bank-Branding.jpg")
    # st.sidebar.image(image)

    # Botão para carregar arquivo na aplicação
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Navegação", type = ['csv','xlsx'])

    # Verifica se há conteúdo carregado na aplicação
    if (data_file_1 is not None):
        df = pd.read_csv(data_file_1)

        st.write('## DataFrame:')
        st.write(df.head())

        variaveis = df.columns.to_list()
        
        st.write('## Distribuição de cada variável:')
        for var in variaveis:
            distribuicao(df, var)
        
        st.write('A maioria das variáveis apresentam um distribuição de decrescimento logarítmico, portanto para aquelas selecionadas será necessário aplicar algum tipo de tratamento para normalização desses dados.')
        
        st.markdown('---')
        
        st.write('## Seleção de variáveis:')
        st.write('Para a criação do cluster com K-Means, vamos considerarar apenas as variáveis que descrevem a características de navegação.')
        st.write('Portanto utilizaremos apenas as seguintes variáveis:\n\n Administrative; \n\n Administrative_Duration; \n\n Informational; \n\n Informational_Duration; \n\n ProductRelated; \n\n ProductRelated_Duration;')
        
        st.write('Para as variáveis selecionadas o tratamento utilizado será yeo-jhonson, para que sua distribuição seja o mais próximo de uma normal.')
        
        df_select = df.iloc[:, :6]
        variaveis = df_select.columns.to_list()
        
        tratamento(df, variaveis)
        
        st.write('As variáveis apresentam distribuições mais próximas da normal.')
        
        st.markdown('---')
        st.write('## Número de grupos:')
        
        st.write('''
                 Para a seleção do número de grupos será utilizados dois métodos, 
                 o do cotovelo que avalia qual a quantidade de clusters que faz com que o SQD sofra uma queda brusca (mais subjetivo),
                 e o da silhueta que avalia qual grupo se ajusta melhor aos dados (mais objetivo).
                 ''')
        
        st.write('### Método do cotovelo:')
        
        pt = PowerTransformer(method='yeo-johnson')
        yeo = df_select.copy()
        yeo[variaveis] = pt.fit_transform(yeo[variaveis])
        
        df_pad = pd.DataFrame(StandardScaler().fit_transform(yeo), columns = yeo.columns)
        
        SQD = []
        K = range(1,15)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(df_pad[variaveis])
            SQD.append(km.inertia_)
            
        df_plot = pd.DataFrame({'num_clusters': list(range(1, len(SQD)+1)), 'SQD': SQD})
        fig = plt.figure(figsize=(10,6))
        ax = sns.lineplot(df_plot, x= 'num_clusters', y='SQD', marker='o')
        ax.set(title='Número de clusters pela soma dos quadrados das distâncias', xlabel='Número de clusters', ylabel='SQD (Soma dos Quadrados das Distâncias)')
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write('A partir do método do cotovelo é possível constatar que o número de clusters que gera a queda mais brusca no SQD é 3.')
        st.write('Como o método do cotovelo é subjetivo, para uma avaliação mais objetiva vamos utilizar o coeficiente de silhueta.')
        
        st.write('### Método da silhueta:')
        
        silhueta(df_pad)
        
        st.write('Como é possível observar acima o número de clusters que aumenta o valor de silhueta média é 3, com o grupo de 4 clusters ficando em segundo lugar.')
        
        km = KMeans(n_clusters=3, random_state=42).fit(df_pad)
        df_pad['grupos_3'] = pd.Categorical(['grupo_' + str(g) for g in km.labels_])
        
        st.markdown('---')
        
        st.write('## Visualização dos grupos criados:')
        
        pca = PCA(n_components=2)
        dados_pca = pca.fit_transform(df_pad[variaveis])
      
        biplot(dados_pca, np.transpose(pca.components_[0:2, :]), df_pad['grupos_3'], labels = df_pad.columns.to_list())
        
        st.write('Os grupos serão renomeados conforme a distribuição do gráfico acima: \n\n grupo_0 = Administrativo; \n\n grupo_1 = Informativo; \n\n grupo_2 = Produtos;')

        st.markdown('---')
        
        st.write('### Avaliação do potencial de compra de cada grupo criado:')
        
        var_final = variaveis + ['grupos_3']

        df_final = df_pad[var_final].copy()
        df_final['grupos_3'] = df_final['grupos_3'].astype(str)
        df_final.loc[df_final['grupos_3'] == 'grupo_0', 'grupos_3'] = 'Administrativo'
        df_final.loc[df_final['grupos_3'] == 'grupo_1', 'grupos_3'] = 'Informativo'
        df_final.loc[df_final['grupos_3'] == 'grupo_2', 'grupos_3'] = 'Produtos'
        
        x = df_final['grupos_3']
        df_all = pd.concat([df, x], axis=1)
        df_all.rename(columns={'grupos_3': 'Grupos'}, inplace=True)
        
        proporcao(df_all)
        
if __name__ == '__main__':
	main()