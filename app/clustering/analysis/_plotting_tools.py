import os
from typing import Any, List, Dict, Union, Optional, Tuple
from itertools import combinations
from matplotlib import gridspec
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
from numpy import count_nonzero
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import sctoolkit.sctoolkit.utils as sctk_utils
from PIL import Image


# ===========================================EDA=====================================================================
def explore_categorical_byGroup(df: pd.DataFrame, 
                                cat_features: List[str], 
                                target_label_column: str, 
                                target_label: str) -> None:
    '''categorical data insight form df

    :param: df, pd dataframe
    :param: cat_features, list or array
    
    :return: plot

    '''
    sns.set_theme(style="whitegrid")

    n_subfigures = len(cat_features)
    n_cols = ceil(n_subfigures**0.5)
    n_rows = int(ceil(n_subfigures / n_cols))

    gs = gridspec.GridSpec(n_rows, n_cols)
    fig = plt.figure(figsize=(20, 8))
    
    fig_count = 0
    df[target_label_column] = df[target_label_column].astype(str)
    
    for cat in cat_features:
        ax = fig.add_subplot(gs[fig_count])
        
        df_grouped = df.groupby(cat)[target_label_column].value_counts().unstack(fill_value=0)
        df_grouped['Total'] = df_grouped.sum(axis=1)
        df_grouped = df_grouped.sort_values(by=['Total'], ascending=False)
        df_grouped.index = df_grouped.index.astype(str)
        
        # Plot the total
        sns.set_color_codes("pastel")
        sns.barplot(x=df_grouped.index, y='Total', data=df_grouped,
                    label="Total", color="b", ax=ax)
        
        # Plot where affluent
        sns.set_color_codes("muted")
        sns.barplot(x=df_grouped.index, y=target_label, data=df_grouped,
                    label=target_label, color="b", ax=ax)
        
        # Add a legend and informative axis label
        ax.legend(ncol=2, loc="upper right", frameon=True)
        ax.set(ylabel="Count",
               xlabel=cat)
        sns.despine(left=True, bottom=True)

        fig_count += 1
      

    plt.tight_layout()
    
    
def explore_categorical(
        df: pd.DataFrame, 
        cat_features=List[str]
    ) -> Any:
    '''
    categorical data insight form df

    :param: df, pd dataframe
    :param: cat_features, list or array
    
    :return: plot

    '''

    n_subfigures = len(cat_features)
    n_cols = ceil(n_subfigures**0.5)
    n_rows = int(ceil(n_subfigures / n_cols))

    gs = gridspec.GridSpec(n_rows, n_cols)
    fig = plt.figure(figsize=(12, 8))
    
    fig_count = 0
    
    for cat in cat_features:
        ax = fig.add_subplot(gs[fig_count])
        sns.barplot(y=df.index, x=cat, data=df, estimator=count_nonzero, ax=ax)
#         ax.set_yscale("log")
        ax.grid(False)
        # ax.xaxis.label.set_visible(False)
#         ax.yaxis.label.set_visible(False)
        ax.title.set_text(cat)

        fig_count += 1

    plt.tight_layout()
    
    

def plot_outlier_graph_set(
        df: pd.DataFrame, 
        numeric_features: List[str]
        ) -> Any:  
    '''categorical data insight form df

    :param: df, pd dataframe
    :param: numberical features, list or array
    
    :return: plot
      '''

    for f in numeric_features:
        fig, axs = plt.subplots(1, 4, figsize=(16, 5))

        # plot 1
        sns.boxplot(y=f, data=df, ax=axs[0])

        # plot 2
        sns.boxenplot(y=f, data=df, ax=axs[1])

        # plot 3
        sns.violinplot(y=f, data=df, ax=axs[2]) 

        # plot 4
        sns.stripplot(y=f, data=df, size=4, color=".3", linewidth=0, ax=axs[3])


        fig.suptitle(f, fontsize=15, y=1.1)
        axs[0].set_title('Box Plot')
        axs[1].set_title('Boxen Plot')
        axs[2].set_title('Violin Plot')
        axs[3].set_title('Strip Plot')

        plt.tight_layout()
        plt.show()


        
def comparison_plot_numerical(
        df_input: pd.DataFrame, 
        numeric_features: List[str], 
        target_label: str, 
        statsmethod: str ='mannwhitneyu',
        figsize: Tuple[int, int] = (10, 10)
        ) -> Any:  
    '''numerical statistical commparison

    :param: df, pd dataframe
    :param: numberical_features, list or array
    :param: target_label, str
    :param: statsmethod, 'mannwhitneyu' or 't_ind'

    :return: plot
      '''
    df = df_input.copy()
    n_subfigures = len(numeric_features)
    n_cols = ceil(n_subfigures**0.5)
    n_rows = int(ceil(n_subfigures / n_cols))

    gs = gridspec.GridSpec(n_rows, n_cols)
    fig = plt.figure(figsize=figsize)

    df[target_label] = df[target_label].astype(str)
    ls_targets = sorted(list(df[target_label].unique()))
    ls_target_combs = [comb for comb in combinations(ls_targets, 2)]   
    dict_label = dict(zip(ls_targets, range(0, len(ls_targets))))

    # Non-FDR
    fig_count = 0
    for feat in numeric_features:
        ax = fig.add_subplot(gs[fig_count])
        df_tmp = df[[feat, target_label]].dropna()
        sns.boxplot(y=feat, x=target_label, data=df_tmp, ax=ax, order=ls_targets)
        # adding signficance bars
        if not statsmethod == 't_ind':   
            for g1, g2 in ls_target_combs:
                v1 = df_tmp[df_tmp[target_label] == g1][feat].values
                v2 = df_tmp[df_tmp[target_label] == g2][feat].values
                t, p = stats.mannwhitneyu(v1, v2)
            
                # statistical annotation
                x1, x2 = dict_label[g1], dict_label[g2]   
                y, h, col = df_tmp[feat].max() + 2, 2, 'k'
                ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
                ax.text((x1+x2)*.5, y+h, "{0:.2e}".format(p), ha='center', va='bottom', color=col, size=12)
                # add log value of median v1 / median v2 to the plot
                ax.text((x1+x2)*.5, y-h, "Effect size:"+"{0:.3f}".format(np.log(np.median(v1)/np.median(v2))), ha='center', va='top', color=col, size=10)


        else:
            for g1, g2 in ls_target_combs:
                v1 = df_tmp[df_tmp[target_label] == g1][feat].values
                v2 = df_tmp[df_tmp[target_label] == g2][feat].values
                t, p = stats.ttest_ind(v1, v2, equal_var=False)
                
                # statistical annotation
                x1, x2 = dict_label[g1], dict_label[g2]   
                y, h, col = df_tmp[feat].max() + 2, 2, 'k'
                ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
                ax.text((x1+x2)*.5, y+h, "{0:.2e}".format(p), ha='center', va='bottom', color=col, size=12)
                # add log value of median v1 / median v2 to the plot
                ax.text((x1+x2)*.5, y-h, "Effect size:"+"{0:.3f}".format(np.log(np.median(v1)/np.median(v2))), ha='center', va='top', color=col, size=10)


        ax.set_yscale("log")
        ax.grid(False)
        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)
        ax.title.set_text(feat)

        fig_count += 1

   
def comparison_plot_categorical(
        df_input: pd.DataFrame,
        cat_features: List[str],
        target_label: str,
        save_path: str='figures/dotplot_compare_cat.png'
    ) -> Any: 
    '''categorical statistical commparison
    '''

    # create figures directory if not exist
    if not os.path.exists('figures'):
        os.makedirs('figures')

    df = df_input.copy()
    df_results = pd.DataFrame()

    # Tabulate the data for each categorical feature
    for cat in cat_features:
        
        # fill in missing for all cat features
        df[cat].fillna('Unknown', inplace=True) 
        df[cat] = df[cat].astype('str')

        df_tab = pd.crosstab(df[cat], df[target_label])
        ls_columns = df_tab.columns

        # get sum values for each column
        sum_1, sum_2 = df_tab.sum(axis=0)

        # get counterpart values for non-target class
        df_tab[f'{ls_columns[0]}_non-target'] = sum_1 - df_tab[f'{ls_columns[0]}']
        df_tab[f'{ls_columns[1]}_non-target'] = sum_2 - df_tab[f'{ls_columns[1]}']

        # perform fisher exact test for each row
        df_tab['Fisher_p'] = df_tab.apply(
            lambda x: stats.fisher_exact(
                [[x[f'{ls_columns[0]}'],
                  x[f'{ls_columns[0]}_non-target']], 
                 [x[f'{ls_columns[1]}'], 
                  x[f'{ls_columns[1]}_non-target']]])[1], 
                  axis=1)
        # logfold change by first calculate the proportion of Affluent over 
        # total Affluent + Affluent_non-target and Normal over total Normal + 
        # Normal_non-target for each row, 
        # then calculate the fold change by proportion of Affluent / proportion of Normal
        df_tab['coefficient'] = df_tab.apply(lambda x: np.log1p((x[f'{ls_columns[0]}'] / \
                                (x[f'{ls_columns[0]}'] + x[f'{ls_columns[0]}_non-target'])) / \
                                 (x[f'{ls_columns[1]}'] / (x[f'{ls_columns[1]}'] + \
                                x[f'{ls_columns[1]}_non-target']))), 
                                axis=1)
        df_tab.reset_index(inplace=True)
        df_tab['Comparison'] = [f'{ls_columns[0]}_vs_{ls_columns[1]}']*df_tab.shape[0]

        # correcting/adjusting pvalue using fdr 
        s, p, _, _ = multipletests(df_tab.Fisher_p, alpha=0.05, method='fdr_bh')
        df_tab['significant'] = s
        df_tab['pval_fdr_adj'] = p
        df_tab['neglog_pval_adj'] = -np.log10(p+1e-50)

        # plot the dotplot
        f = sctk_utils.plot_significance_dotplot(df=df_tab, fill_limit=(-2,2), width_scale=1.0, height_scale=1.2, 
                            size_limit=20, dot_size_limit=20, xcol='Comparison', ycol=cat,
                            xlabel=cat, ylabel='Classes')
        f.save(f'figures/dotplot_{cat}.png')

        df_results = pd.concat([df_results, df_tab], axis=0)


    n_subfigures = len(cat_features)

    # Now, use PIL to open and concatenate the images
    _image = Image.open(f'figures/dotplot_{cat_features[0]}.png')  

    # Assuming the images have the same width
    new_image = Image.new('RGB', (_image.width * n_subfigures, 
                                  _image.height))


    # append image horizontally and vertically
    for cat, i in zip(cat_features, range(len(cat_features))):
        _image = Image.open(f'figures/dotplot_{cat}.png')  
        new_image.paste(_image, (_image.width * i ,0))
    
    new_image.save(save_path)

    # remove all the temporary images
    for cat in cat_features:
        os.remove(f'figures/dotplot_{cat}.png')
    
    ls_default_col =  ['AFFLUENT', 'NORMAL', 'AFFLUENT_non-target',
       'NORMAL_non-target', 'Fisher_p', 'coefficient', 'Comparison',
       'significant', 'pval_fdr_adj', 'neglog_pval_adj']

    return df_results[cat_features + ls_default_col], new_image





