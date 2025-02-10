import pandas as pd
from scipy.stats import mannwhitneyu

# Cargar los DataFrames
df1 = pd.read_csv("outputs/geolife_transformer_default_1738876532.csv")
df2 = pd.read_csv("outputs/geolife_transformer_default_1738880109.csv")

df1 = df1.iloc[:-16]
df2 = df2.iloc[:-16]

# Función para calcular media y desviación estándar
def get_avg_std(df, label):
    df_filtered = df.drop(columns=['type'])
    mean_values = df_filtered.mean()
    std_values = df_filtered.std()
    return mean_values, std_values

# Calcular estadísticas para df1
df1_test_avg, df1_test_std = get_avg_std(df1[df1['type'] == 'test'], "GEOLIFE-Vanilla-TEST")
df1_vali_avg, df1_vali_std = get_avg_std(df1[df1['type'] == 'vali'], "GEOLIFE-Vanilla-VALI")

# Calcular estadísticas para df2
df2_test_avg, df2_test_std = get_avg_std(df2[df2['type'] == 'test'], "GEOLIFE-Sentiments-TEST")
df2_vali_avg, df2_vali_std = get_avg_std(df2[df2['type'] == 'vali'], "GEOLIFE-Sentiments-VALI")

# Crear DataFrame con los resultados
result_df = pd.DataFrame({
    "GEOLIFE-Vanilla-TEST-Mean": df1_test_avg,
    "GEOLIFE-Vanilla-TEST-Std": df1_test_std,
    "GEOLIFE-Sentiments-TEST-Mean": df2_test_avg,
    "GEOLIFE-Sentiments-TEST-Std": df2_test_std
})



# Mostrar resultados
print(result_df)

# Prueba de Mann–Whitney U
p_values = {}
for col in df1.columns:
    if col != "type" and col != "correct@1" and col != "correct@5" and col != "correct@10" and col != "total" and col!="rr" and col!="correct@3":
        stat_test, p_test = mannwhitneyu(df1[df1['type'] == 'test'][col], df2[df2['type'] == 'test'][col], alternative='two-sided')
        #stat_vali, p_vali = mannwhitneyu(df1[df1['type'] == 'vali'][col], df2[df2['type'] == 'vali'][col], alternative='two-sided')
        p_values[col] = {"TEST p-value": p_test}#, "VALI p-value": p_vali}

# Convertir resultados a DataFrame
p_values_df = pd.DataFrame(p_values).T

# Mostrar los p-values
print("\nMann–Whitney U test results:")
print(p_values_df)
