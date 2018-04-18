import nltk
import math
from nltk.corpus import brown
from sklearn.metrics import accuracy_score

l = [ ]
t = len(brown.tagged_sents())*float(0.8)
t = int(t)
ll = brown.tagged_sents()
train = ll[:t]
test = ll[t:]	
train= train[:1000]

for line in train:
	#print line
	l.append( ("<s>", "<s>") )
	l.extend([ (word,tag[:2]) for (word,tag) in line])
	l.append(("</s>","</s>"))
	#print l
	#break

#print l


freqdist = nltk.ConditionalFreqDist(l)
probdist = nltk.ConditionalProbDist(freqdist, nltk.MLEProbDist)


tags = [tag for (word,tag) in l]

freqdist_tags= nltk.ConditionalFreqDist(nltk.bigrams(tags))
probdist_tags = nltk.ConditionalProbDist(freqdist_tags, nltk.MLEProbDist)


actual = []
predicted =[]

tagset = set(tags)
#print tagset
#print len(tagset)
test = test[:200]
for sent in test:
	#actual = []
	#predicted =[]
 	sentence = [word for (word,tag) in sent]
 	actual.extend([tag[:2] for (word,tag) in sent])
 	#print sentence
# #sentence = ["He", "is", "a", "nice", "guy" ]
	path = {"0 <s>":""}
	score = {}
	for tag in tagset:
		key = "1 " + tag
		if tag=="<s>" or tag=="</s>":
			continue
		p1 = probdist_tags['<s>'].prob(tag)
		if p1==0:
			p1 = 1/float(len(tags))
		p2 = probdist[tag].prob(sentence[0])
		if p2==0:
			p2 = 1/float(len(l))	
		temp = (-1)*math.log(p1) + (-1)*math.log(p2)
		score[key] = temp
		path[key] = "0 <s>"

	#print score	

	for i in range(2,len(sentence)+1):
		for tag in tagset:
			key = str(i) + " " + tag
			if tag=="<s>" or tag=="</s>":
				continue
			mini = 1000000000000000.0
			for innertags in tagset:
				if innertags=="<s>" or innertags=="</s>":
					continue
				key1 = str(i-1)
				key1 = key1 + " " + innertags
				temp = 0
				if score.has_key(key1):
					temp = score[key1]
				p1 = probdist_tags[innertags].prob(tag)
				if p1==0:
					p1 = 1/float(len(tags))
				p2 = probdist[tag].prob(sentence[i-1])
				if p2==0:
					p2 = 1/float(len(l))
				temp +=  (-1)*math.log(p1) + (-1)*math.log(p2)
				if mini > temp:
					mini = temp
					temppath = key1
			score[key] = mini
			path[key] = temppath				 
	#print score

	mini = 1000000000000.0
	key = str(len(sentence)+1) + " " + "</s>"
	for tag in tagset:
		if tag=="<s>" or tag=="</s>":
			continue
		temp = score[str(len(sentence)) + " " + tag]
		p1 = probdist_tags["</s>"].prob(tag)
		if p1==0:
			p1 = 1/float(len(tags))
		temp += (-1)*math.log(p1)
		if mini > temp:
			mini = temp
			temppath = str(len(sentence)) + " " + tag

	score[key] = mini
	path[key] = temppath

	#print path.keys()
	nextedge = path[key]
	#print nextedge
	#print path[nextedge]

	tags = []

	while nextedge!="0 <s>":
		#print nextedge		
		tag = nextedge.split()[1]
		tags.append(tag)
		nextedge = path[nextedge]
	tags.reverse()
	predicted.extend(tags)
print accuracy_score(actual, predicted)

	#print predicted
	#print actual
	#print sent
	#break	

