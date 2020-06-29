def create_new_column_name(col_name, suffix, dataset_columns):
    """
    input:
        - col_name: column name of the dataset df
        - suffix: suffix to be appended to col_name
        - dataset_columns: list of all columns of dataset
    output:
        - new column name with suffix that is not in the columns of the dataset
    To ensure the output column name is not in the dataset, we append a number to col_name + "_" + suffix if needed
    """
    if col_name + "_" + suffix in dataset_columns:
        i = 0
        col_new_name = col_name + "_" + suffix + "_" + str(i)

        while col_new_name in dataset_columns:
            i += 1
            col_new_name = col_name + "_" + suffix + "_" + str(i)
        return col_new_name
    else:
        return col_name + "_" + suffix
    
    
def create_all_new_column_names(df_columns, text_col_list):
    """
    All new column names creation are stored in the three following dictionaries 
    - col_txt_lang_dict[col_txt_name]: col_lang_name
    - col_txt_preprocessed_dict[col_txt_name]: col_preprocessed_name 
    - col_txt_spellchecked_dict[col_txt_name]: col_no_typos_name
    """

    col_txt_lang_dict = {}
    col_txt_preprocessed_dict = {}
    col_txt_spellchecked_dict = {}

    df_all_columns = list(df_columns)

    for col_txt_name in text_col_list:
        new_col = create_new_column_name(col_txt_name, "lang", df_all_columns)
        col_txt_lang_dict[col_txt_name] = new_col
        df_all_columns.append(new_col)

        new_col = create_new_column_name(col_txt_name, "preprocessed", df_all_columns)
        col_txt_preprocessed_dict[col_txt_name] = new_col
        df_all_columns.append(new_col)

        new_col = create_new_column_name(col_txt_name, "no_typos", df_all_columns)
        col_txt_spellchecked_dict[col_txt_name] = new_col
        df_all_columns.append(new_col)

    return df_all_columns, col_txt_lang_dict, col_txt_preprocessed_dict, col_txt_spellchecked_dict