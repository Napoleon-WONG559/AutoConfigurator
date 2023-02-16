from urllib.request import urlopen
import re
import pandas as pd

#target website for data crawling
link = "https://benchmarks.ul.com/compare/best-gpus"

f = urlopen(link)
myfile = f.read()

#target string for matching
link_regex="https://benchmarks.ul.com/hardware/gpu/[A-Za-z' '0-9]*"
link=re.findall(link_regex,str(myfile))

#extract the graphic card name from the links
rank_result=[]
for item in link:
    rank_result.append(item[39:])
print(rank_result)

#output to a xlsx file
df1 = pd.DataFrame(rank_result,
                   columns=['graphicard'])
df1.to_excel("output_graphicard_rank.xlsx")