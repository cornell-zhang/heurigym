import sys
import pytoulbar2

class Data:
	def __init__(self, filename, k):
		self.var = {}
		self.dom = {}
		self.ctr = list()
		self.softeq = list()
		self.softne = list()
		self.nbsoft = 0

		stream = open(filename)
		for line in stream:
			if len(line.split())==3 and line.split()[0]=="DM":
				(DM, dom, freq) = line.split()[:3]
				if self.dom.get(int(dom)) is None:
					self.dom[int(dom)] = [int(freq)]
				else:
					self.dom[int(dom)].append(int(freq))

			if len(line.split()) == 4 and line.split()[0]=="TR":
				(TR, route, dom, polarisation) = line.split()[:4]
				if int(polarisation) == 0:
					self.var[int(route)] = [(f,-1) for f in self.dom[int(dom)]] + [(f,1) for f in self.dom[int(dom)]]
				if int(polarisation) == -1:
					self.var[int(route)] = [(f,-1) for f in self.dom[int(dom)]] 
				if int(polarisation) == 1:
					self.var[int(route)] = [(f,1) for f in self.dom[int(dom)]]

			if len(line.split())==6 and line.split()[0]=="CI":
				(CI, route1, route2, vartype, operator, deviation) = line.split()[:6]
				self.ctr.append((int(route1), int(route2), vartype, operator, int(deviation)))

			if len(line.split())==14 and line.split()[0]=="CE":
				(CE, route1, route2, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10) = line.split()[:14]
				self.softeq.append((int(route1), int(route2), [int(s) for s in [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]]))
				self.nbsoft += 1

			if len(line.split())==14 and line.split()[0]=="CD":
				(CD, route1, route2, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10) = line.split()[:14]
				self.softne.append((int(route1), int(route2), [int(s) for s in [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]]))

		self.top = 10*(k+1)*self.nbsoft**2 + 1        

if len(sys.argv) < 2:
	exit('Command line argument is composed of the problem data filename and the relaxation level')
k = int(sys.argv[2])

#collect data
data = Data(sys.argv[1], k)

Problem = pytoulbar2.CFN(data.top)

#create a variable for each link
for e in list(data.var.keys()):
	domain = []
	for i in data.var[e]:
		domain.append(str(i))
	Problem.AddVariable("X" + str(e), domain)
	
#hard binary constraints
for (route1, route2, vartype, operand, deviation) in data.ctr:
	Constraint = []
	for (f1,p1) in data.var[route1]:
	 	for (f2,p2) in data.var[route2]:
	 		if vartype == 'F':
	 			if operand == 'E':
	 				if abs(f2 - f1) == deviation:
	 					Constraint.append(0)
	 				else:
	 					Constraint.append(data.top)
	 			else:
	 				if abs(f2 - f1) != deviation:
	 					Constraint.append(0)
	 				else:
	 					Constraint.append(data.top)
	 		else:
	 			if operand == 'E':
	 				if p2 == p1:
	 					Constraint.append(0)
	 				else:
	 					Constraint.append(data.top)
	 			else:
	 				if p2 != p1:
	 					Constraint.append(0)
	 				else:
	 					Constraint.append(data.top)
	Problem.AddFunction(["X" + str(route1), "X" + str(route2)], Constraint)


#soft binary constraints for equal polarization
for (route1, route2, deviations) in data.softeq:
	for i in range(11):
		ListConstraints = []
		for (f1,p1) in data.var[route1]:
			for (f2,p2) in data.var[route2]:
	 			if p1!=p2 or abs(f1 - f2) >= deviations[i]:
	 				ListConstraints.append(0)
	 			elif i >= k:
	 				ListConstraints.append(data.top)
	 			elif i == k-1:
	 				ListConstraints.append(10*data.nbsoft)
	 			else:
	 				ListConstraints.append(1)
		Problem.AddFunction(["X" + str(route1), "X" + str(route2)], ListConstraints)


#soft binary constraints for not equal polarization
for (route1, route2, deviations) in data.softne:
	for i in range(11):
		ListConstraints = []
		for (f1,p1) in data.var[route1]:
	 		for (f2,p2) in data.var[route2]:
	 			if p1==p2 or abs(f1 - f2) >= deviations[i]:
	 				ListConstraints.append(0)
	 			elif i >= k:
	 				ListConstraints.append(data.top)
	 			elif i == k-1:
	 				ListConstraints.append(10*data.nbsoft)
	 			else:
	 				ListConstraints.append(1)
		Problem.AddFunction(["X" + str(route1), "X" + str(route2)], ListConstraints)

#zero-arity cost function representing a constant cost corresponding to the relaxation at level k
Problem.AddFunction([], 10*k*data.nbsoft**2)

#Problem.Dump('Fapp.cfn')
Problem.CFN.timer(900)
Problem.Solve(showSolutions=3)

