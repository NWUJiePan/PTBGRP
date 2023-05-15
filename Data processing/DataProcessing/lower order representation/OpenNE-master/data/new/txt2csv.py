#  @author rzh
#  @create 2021-04-25 20:41

input_file = open('phage host pairs988.csv', 'r')
output_file = open('phagehost_pairs988.txt', 'w')
# input_file.readline() # skip first line
for line in input_file:
    output_file.write(' '.join(line.strip().split(',')) + '\n')
input_file.close()
output_file.close()