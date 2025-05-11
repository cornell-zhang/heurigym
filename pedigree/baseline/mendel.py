import sys
import pytoulbar2

class Data:
	def __init__(self, ped):
		self.id = list()
		self.father = {}
		self.mother = {}
		self.allelesId = {}
		self.ListAlle = list()
		self.obs = 0

		stream = open(ped)
		for line in stream:
			(locus, id, father, mother, sex, allele1, allele2) = line.split()[:]
			self.id.append(int(id))
			self.father[int(id)] = int(father)
			self.mother[int(id)] = int(mother)
			self.allelesId[int(id)] = (int(allele1), int(allele2)) if int(allele1) < int(allele2) else (int(allele2), int(allele1))
			if not(int(allele1) in self.ListAlle) and int(allele1) != 0:
				self.ListAlle.append(int(allele1))
			if int(allele2) != 0 and not(int(allele2) in self.ListAlle):
				self.ListAlle.append(int(allele2))
			if int(allele1) != 0 or int(allele2) != 0:
				self.obs += 1

#collect data
data = Data(sys.argv[1])
top = int(data.obs+1)

Problem = pytoulbar2.CFN(top)

#create a variable for each individual
for i in data.id:
	domains = []
	for a1 in data.ListAlle:
		for a2 in data.ListAlle:
			if a1 <= a2:
				domains.append('a'+str(a1)+'a'+str(a2))
	Problem.AddVariable('g' + str(i) , domains)

#create the constraints that represent the mendel's laws
ListConstraintsMendelLaw = []
for p1 in data.ListAlle:
	for p2 in data.ListAlle:
		if p1 <= p2:	# father alleles
			for m1 in data.ListAlle:
				for m2 in data.ListAlle:
					if m1 <= m2:	# mother alleles
						for a1 in data.ListAlle:
							for a2 in data.ListAlle:
								if a1 <= a2:	# child alleles
									if (a1 in (p1,p2) and a2 in (m1,m2)) or (a2 in (p1,p2) and a1 in (m1,m2)) :
										ListConstraintsMendelLaw.append(0)
									else :
									 	ListConstraintsMendelLaw.append(top)

for i in data.id:
	#ternary constraints representing mendel's laws
	if data.father.get(i, 0) != 0 and data.mother.get(i, 0) != 0:
		Problem.AddFunction(['g' + str(data.father[i]),'g' + str( data.mother[i]), 'g' + str(i)], ListConstraintsMendelLaw)
		
	#unary constraints linked to the observations
	if data.allelesId[i][0] != 0 and data.allelesId[i][1] != 0:
		ListConstraintsObservation = []
		for a1 in data.ListAlle:
			for a2 in data.ListAlle:
				if a1 <= a2:
					if (a1,a2) == data.allelesId[i]:
						ListConstraintsObservation.append(0)
					else :
					 	ListConstraintsObservation.append(1)
		Problem.AddFunction(['g' + str(i)], ListConstraintsObservation)

#Problem.Dump('Mendel.cfn')
Problem.CFN.timer(300)
res = Problem.Solve(showSolutions=2)
if res:
	print('COST:',int(res[1]))
else:
	print('No solution found')

