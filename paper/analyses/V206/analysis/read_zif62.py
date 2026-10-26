import pandas as pd
p = r'D:/Downloads/材料agent/PATH-Mat_Evidence_Strengthening_V184/inputs/ZIF62_Order_Supplementary_SourceData.xlsx'
x = pd.ExcelFile(p)
print(x.sheet_names)
for s in x.sheet_names:
 print('---', s)
 print(pd.read_excel(p, sheet_name=s).to_string(index=False)[:10000])
