import pandas as pd
import numpy as np
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import linregress as lrs

import hw2_regression as reg


def filtering(genotypes, strains):
    relevant_strains = [column for column in genotypes.columns if column in strains]
    columns = list(genotypes.columns)[:3]
    columns.extend(relevant_strains)
    final_genotypes = genotypes[columns]
    drop_indices = []
    for i in range(len(final_genotypes) - 1):
        if (final_genotypes.iloc[i, 3:]).equals(final_genotypes.iloc[i + 1, 3:]):
            drop_indices.append(i + 1)
    final_genotypes = final_genotypes.drop(drop_indices)
    return final_genotypes


def mean_data(table):
    cols_set = {col.split('_')[0] for col in table.columns if 'BXD' in col}
    means_table = pd.DataFrame({'data': table['data']})
    for strain in cols_set:
        temp_cols = [col for col in table.columns if col.split('_')[0] == strain]
        means_table[strain] = table[temp_cols].mean(axis=1)
    sorted_cols = list(table.columns[:1])
    sorted_cols.extend(sorted(means_table.columns[1:], key=lambda x: int(x[3:])))
    means_table = means_table[sorted_cols]
    return means_table


def association_model(genotypes_df, expression_df, alpha=0.05):
    relevant_cols = [col for col in genotypes_df.columns if col in expression_df.columns]
    genotypes_df = genotypes_df[genotypes_df.columns[:3].to_list() + relevant_cols]
    strains_dict = dict()
    for i in range(len(expression_df)):
        p_vals_list = []
        phe = expression_df.iloc[i]
        for index in genotypes_df.index:
            gen = genotypes_df.loc[index]
            non_others_indexes = gen[(((gen == 'B') | (gen == 'D')) | (gen == 'H'))].index
            gen, phe = gen[non_others_indexes], phe[non_others_indexes].astype(float)
            gen = gen.map({'B': 2, 'H': 1, 'D': 0}).astype(int)
            p_vals_list.append(lrs(gen, phe)[3])
        strains_dict[phe['data']] = p_vals_list
    raw_mat = pd.DataFrame(strains_dict)
    raw_mat = raw_mat.T
    raw_mat.index = expression_df['data']
    raw_mat.columns = genotypes_df.index
    stacked = raw_mat.stack()
    stacked = pd.Series(fdrcorrection(stacked, 0.05)[1], index=stacked.index)
    final_mat = stacked.unstack()
    final_mat.reset_index(inplace=True)
    final_mat['min_p_val'] = final_mat.min(axis=1)
    final_mat = final_mat[final_mat['min_p_val'] < alpha]
    final_mat.drop('min_p_val', axis='columns', inplace=True)
    return final_mat


if __name__ == '__main__':
    genotypes_file = pd.read_excel('genotypes.xls', skiprows=1).drop(0, axis=1)
    txt_file = "dendritic_PAM_stimulation.txt"
    data = pd.read_csv(txt_file, sep='\t', header=0)
    strains = mean_data(data)
    genotypes = filtering(genotypes_file, strains.columns[1:])

    # p_vals = association_model(genotypes, strains)  ## Takes a long time, csv attached and imported in the line below:
    p_vals = pd.read_csv("after association test.csv")

    locations_file = "MGI_Coordinates.Build37.rpt.txt"
    locations = pd.read_csv(locations_file, sep='\t', header=0)
