import argparse
class Solution(object):
	def __init__(self, filename):
		self.filename = filename
	def fill_triag(self):
		out = []
		with open(self.filename, 'r') as f:
			num_row = 0
			for line in f:
				row = list(map(int, list(filter(None, line.rstrip().split(" ")))))
				num_row += 1
				out.append(row)
			print(out[:5])
			for row in out:
				add = [float("inf") for i in range(num_row-len(row))]
				row += add
		return out
	def find_min_path(self):
		mat = self.fill_triag()
		out = [mat[0][0]]
		cost = mat[0][0]
		i = 0
		j = 0
		while i < len(mat)-1 and j < len(mat[0])-1:
			if i < 6:
				print(cost)
			left = mat[i+1][j]
			right = mat[i+1][j+1]
			if left < right:
				cost += left
				i += 1
				out.append(left)
			else:
				cost += right
				i += 1
				j += 1
				out.append(right)
		print(cost)
		return out

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('filename', help='Directory containing full dataset')
	args = parser.parse_args()
	sol = Solution(args.filename)
	return sol.find_min_path()

if __name__ == '__main__':
	res = main()
	for r in res[:10]:
		print(r)