textFile = open('result.txt', "r")
lines = textFile.readlines()
count = 0
for line in lines:
    if int(line.split(" ")[1][4]) is 1:
        count+=1
print(count)
# with open('result2.txt', 'a') as the_file:
#     for f in lines:
#         the_file.write(f.split(" ")[0] + ' ' + '0000' + str(f.split(" ")[1][3]) + '0\n')
