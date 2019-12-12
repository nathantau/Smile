# zipped_coordinates = [(1,2), (3,4), (5,6)]

# unzipped_coordinates = [item for tuple in zipped_coordinates for item in tuple]

# print(unzipped_coordinates)

zipped_coordinates = [(1,2),(3,4)]

unzipped_coordinates = list(sum(zipped_coordinates, ()))

print(unzipped_coordinates)