import argparse
class Solution(object):
	def __init__(self, filename):
		self.filename = filename
		self.res = []
	
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
		self.helper(0, 0, mat)
		return self.res

	def helper(self, r, c, mat):
		if r >= len(mat) or c >= len(mat[0]):
			return mat[r][c]
		min_cost = min(self.helper(r+1, c, mat), self.helper(r+1, c+1, mat))
		mat[r][c] = min_cost
		self.res.append(min_cost)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('filename', help='Directory containing full dataset')
	args = parser.parse_args()
	sol = Solution(args.filename)
	return sol.find_min_path()

if __name__ == '__main__':
	res = main()
	for r in res:
		print(r)