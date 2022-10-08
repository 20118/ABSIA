import pandas as pd 
df=pd.read_csv("train.csv")

tweets=list(df.text.values)
asps=list(df.aspects.values)
senti=list(df.ordinal.values)
pol=list(df.sentiment.values)

for i in range(len(senti)):
	if(senti[i]==0):
		senti[i]="neutral"
	elif(senti[i]==-1):
		senti[i]="low negative"
	elif(senti[i]==-2):
		senti[i]="moderate negative"
	elif(senti[i]==-3):
		senti[i]="high negative"
	elif(senti[i]==1):
		senti[i]="low positive"
	elif(senti[i]==2):
		senti[i]="moderate positive"
	elif(senti[i]==3):
		senti[i]="high positive"


fr=open("train.txt","w")
for j in range(len(df)):
	sent= tweets[j]
	asp=asps[j]
	
	target_temp1="The sentiment polarity of the aspect "+asp+" is "+pol[j]+ " ." 
	target_temp2="The aspect "+asp+" has "+senti[j]+" intensity."
	
	fr.write(str(str(sent)));fr.write("\t")
	fr.write(str(target_temp1));fr.write("\t")
	fr.write(str("pol"));
	fr.write("\n")
	fr.write(str(str(sent)));fr.write("\t")
	fr.write(str(target_temp2));fr.write("\t")
	fr.write(str("intt"))
	fr.write("\n")







	
